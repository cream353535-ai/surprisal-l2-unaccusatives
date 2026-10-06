"""compute_surprisal.py — word-level surprisal and contextual entropy for the 32 target sentences.

Procedure (as described in the paper, Section 3.4):
  * Models: GPT-2 Small/Medium/Large and SmolLM2 135M/360M/1.7B, pretrained base checkpoints.
  * Each sentence is fed verbatim (original capitalization and punctuation), with NO instruction
    and NO beginning-of-sequence token.
  * Token surprisal  S(t_j) = -log2 P(t_j | t_1 ... t_{j-1}).
  * Word surprisal   S(w_i) = sum of the surprisals of the tokens that make up w_i.
  * Contextual entropy for word w_i = Shannon entropy (bits) of the next-token distribution over the
    full vocabulary immediately before the FIRST token of w_i.
  * Tokens are aligned to whitespace-delimited words with the tokenizer's offset mapping.
  * The first word ("As") has no preceding context: surprisal and entropy are NaN.

Input : data/stimuli.csv          (item, pair, condition, animacy, alternation, verb, sentence)
Output: data/surprisal_long.csv   (one row per model x sentence x word)

Usage:
    python scripts/compute_surprisal.py                # all six models
    python scripts/compute_surprisal.py --models gpt2  # a subset
    python scripts/compute_surprisal.py --device cuda
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]

# key used in the output file -> Hugging Face checkpoint
MODELS = {
    "gpt2-small":   "openai-community/gpt2",
    "gpt2-medium":  "openai-community/gpt2-medium",
    "gpt2-large":   "openai-community/gpt2-large",
    "smollm2-135m": "HuggingFaceTB/SmolLM2-135M",
    "smollm2-360m": "HuggingFaceTB/SmolLM2-360M",
    "smollm2-1.7b": "HuggingFaceTB/SmolLM2-1.7B",
}

# Word positions (1-based) analysed in the paper. All target sentences share the frame
# As(1) the(2) NP1(3) Verb(4) Det(5) NP2(6) was(7) being(8) PastPart(9) ...
POSITION_LABEL = {1: "As", 2: "the", 3: "NP1", 4: "Verb", 5: "Det", 6: "NP2",
                  7: "was", 8: "being", 9: "PastPart"}


def word_spans(sentence: str):
    """Character spans (start, end) of whitespace-delimited words."""
    spans, start = [], None
    for i, ch in enumerate(sentence):
        if ch != " " and start is None:
            start = i
        if ch == " " and start is not None:
            spans.append((start, i)); start = None
    if start is not None:
        spans.append((start, len(sentence)))
    return spans


def score_sentence(sentence, tokenizer, model, device):
    enc = tokenizer(sentence, return_offsets_mapping=True, add_special_tokens=False,
                    return_tensors="pt")
    ids = enc["input_ids"].to(device)
    offsets = enc["offset_mapping"][0].tolist()

    with torch.no_grad():
        logits = model(ids).logits[0]                      # (n_tokens, vocab)
    logprobs = torch.log_softmax(logits.float(), dim=-1)   # natural log
    LOG2E = 1.0 / math.log(2)

    n_tok = ids.shape[1]
    # surprisal of token j (bits) given tokens < j; token 0 has no context
    tok_surprisal = [float("nan")] + [
        (-logprobs[j - 1, ids[0, j]].item()) * LOG2E for j in range(1, n_tok)
    ]
    # entropy of the distribution predicting token j (bits), i.e. computed from logits at j-1
    probs = logprobs.exp()
    ent = -(probs * logprobs).sum(-1) * LOG2E              # entropy at each context position
    tok_entropy_before = [float("nan")] + [ent[j - 1].item() for j in range(1, n_tok)]

    # align tokens to words: a token belongs to the word containing its last character
    spans = word_spans(sentence)
    words = [sentence[s:e] for s, e in spans]
    tok2word = []
    for (s, e) in offsets:
        last = max(e - 1, s)
        wid = next((k for k, (ws, we) in enumerate(spans) if ws <= last < we), None)
        tok2word.append(wid)

    rows = []
    for wi, word in enumerate(words):
        toks = [j for j, w in enumerate(tok2word) if w == wi]
        if not toks:
            raise RuntimeError(f"No tokens aligned to word {word!r} in: {sentence}")
        first = toks[0]
        surp = float("nan") if wi == 0 else sum(tok_surprisal[j] for j in toks)
        entr = float("nan") if wi == 0 else tok_entropy_before[first]
        rows.append(dict(wpos=wi + 1, word=word, n_tokens=len(toks),
                         word_length=len(word.rstrip(".,;:!?")),
                         surprisal=surp, entropy=entr))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(MODELS), help="subset of model keys")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--stimuli", default=ROOT / "data" / "stimuli.csv")
    ap.add_argument("--out", default=ROOT / "data" / "surprisal_long.csv")
    args = ap.parse_args()

    stim = pd.read_csv(args.stimuli)
    out = []
    for key in args.models:
        ckpt = MODELS[key]
        print(f"[{key}] loading {ckpt} on {args.device}")
        tok = AutoTokenizer.from_pretrained(ckpt)
        model = AutoModelForCausalLM.from_pretrained(ckpt).to(args.device).eval()
        for _, r in stim.iterrows():
            for row in score_sentence(r.sentence, tok, model, args.device):
                out.append(dict(model=key, item=r.item, pair=r.pair, condition=r.condition,
                                animacy=r.animacy, alternation=r.alternation, **row,
                                position=POSITION_LABEL.get(row["wpos"], f"w{row['wpos']}")))
        del model; torch.cuda.empty_cache() if args.device.startswith("cuda") else None

    df = pd.DataFrame(out)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}  ({len(df)} rows)")

    # quick check: mean surprisal by condition at positions 4-8, averaged over models
    chk = (df.query("4 <= wpos <= 8").groupby(["condition", "position"])["surprisal"]
             .mean().unstack()[["Verb", "Det", "NP2", "was", "being"]])
    print(chk.round(2))


if __name__ == "__main__":
    main()

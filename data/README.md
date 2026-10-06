# Data files

## stimuli.csv
| column | description |
|---|---|
| item | sentence number 1–32 (1–16 inanimate subject, 17–32 animate subject) |
| pair | sentence-pair id 1–16; items *k* and *k*+16 share everything except NP1 |
| condition | AA, IA, AN, IN (first letter: Animate/Inanimate subject; second: Alternating/Non-alternating verb) |
| animacy | animate / inanimate |
| alternation | alternating / non-alternating |
| verb | subordinate-clause verb (past tense) |
| sentence | full sentence as presented |

## human_rt_trials_pos4-8.csv
One row per participant × sentence × word position (positions 4–8 only). Raw (untrimmed) reading times.
| column | description |
|---|---|
| subject | anonymous participant id |
| item | sentence number (joins to `stimuli.csv`) |
| condition | AA / IA / AN / IN |
| position | Verb, Det, NP2, was, being |
| wpos | word position in the sentence (4–8) |
| rt | reading time in ms |

No personal information is included. Trimming (100–3000 ms) is applied in the analysis scripts, not in this file.

## surprisal_long.csv
One row per model × sentence × word. Produced by `scripts/compute_surprisal.py`.
| column | description |
|---|---|
| model | gpt2-small, gpt2-medium, gpt2-large, smollm2-135m, smollm2-360m, smollm2-1.7b |
| item, pair, condition, animacy, alternation | as in `stimuli.csv` |
| wpos | word position (1 = *As*) |
| position | label: As, the, NP1, Verb, Det, NP2, was, being, PastPart, w10… |
| word | word string (punctuation attached as presented) |
| n_tokens | number of sub-word tokens the word was split into |
| word_length | characters, excluding final punctuation |
| surprisal | word surprisal in bits (sum over tokens); NaN for *As* |
| entropy | entropy in bits of the next-token distribution immediately before the word's first token; NaN for *As* |

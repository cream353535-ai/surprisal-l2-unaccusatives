# Language-Model Surprisal and Korean L2 Sentence Processing: Animacy and Unaccusative Verb Alternation

Data, stimuli, and analysis code for:

> *Predicting Korean L2 Sentence Processing with Language-Model Surprisal: The Roles of Animacy and Unaccusative Verb Alternation.* (manuscript under review)

The study compares word-by-word self-paced reading times from 71 Korean learners of English with
word-level surprisal and contextual entropy from six language models (GPT-2 Small/Medium/Large,
SmolLM2 135M/360M/1.7B) on 32 English garden-path sentences crossing **subject animacy**
(animate / inanimate) and **unaccusative verb alternation** (alternating / non-alternating), e.g.

> *As the thief broke the sculpture was being painted in the strange arena.*

## Repository layout

```
data/
  stimuli.csv                   32 target sentences with condition coding (from Nelson et al., 2021, Exp. 1)
  human_rt_trials_pos4-8.csv    trial-level reading times at word positions 4-8 (Verb, Det, NP2, was, being)
  surprisal_long.csv            word-level surprisal and entropy for all six models (output of compute_surprisal.py)
scripts/
  compute_surprisal.py          computes word-level surprisal and contextual entropy from the Hugging Face checkpoints
  analyze_surprisal.R           all statistical analyses reported in the paper (lme4)
  make_figures.py               Figure 1(a), 1(b) and Appendix Figure A1
results/                        CSV tables written by analyze_surprisal.R
figures/                        PDF/PNG figures written by make_figures.py
```

Column definitions are given in `data/README.md`.

## Reproducing the analyses

```bash
# 1. Python environment (Python >= 3.9)
pip install -r requirements.txt

# 2. Language-model measures (downloads ~4 GB of checkpoints on first run; CPU is fine, GPU is faster)
python scripts/compute_surprisal.py            # -> data/surprisal_long.csv
#   python scripts/compute_surprisal.py --models gpt2-small smollm2-135m   # quick test

# 3. Statistics (R >= 4.1 with dplyr, tidyr, lme4)
Rscript scripts/analyze_surprisal.R            # -> results/*.csv

# 4. Figures
python scripts/make_figures.py                 # -> figures/*.pdf, *.png
```

`data/surprisal_long.csv` is included, so steps 3 and 4 can be run without step 2.

## Method summary

* **Surprisal.** Each sentence is fed verbatim (no instruction, no BOS token). Token surprisal is
  −log₂ P(tₖ | t₁…tₖ₋₁); a word's surprisal is the sum over its sub-word tokens.
* **Contextual entropy.** Shannon entropy (bits) of the next-token distribution over the full
  vocabulary immediately before the first token of each word.
* **Reading-time preprocessing.** Trials with RT < 100 ms or > 3000 ms are excluded; analyses use
  natural-log RT at five positions (Verb, Det, NP2, *was*, *being*).
* **Statistics.** Mixed-effects models (lme4, ML, random intercepts for participants and the 16
  sentence pairs) with likelihood-ratio tests; by-participant and item-level *t* tests by position;
  Pearson correlations between sentence–position means and surprisal; trial-level comparison of a
  baseline model against models adding standardized surprisal and/or entropy; separate *was* / *being*
  models including the surprisal of the current and the preceding word.

Which result table corresponds to which table/figure in the paper:

| Paper | File |
|---|---|
| Table 2 (RT means) | `results/02_rt_means_by_condition_position.csv` |
| Section 4.1 omnibus LMM | `results/03_omnibus_lmm_lrt.csv` |
| Table 3 (position effects) | `results/04a_…`, `04b_…`, `04c_…` |
| Tables 4, 5, A1 (surprisal / entropy by condition) | `results/05a_…`, `05b_…`, `05c_…` |
| Table 6 (correlations) | `results/06_correlations_rt_vs_lm_measures.csv` |
| Table 7 (baseline vs. augmented) | `results/07_trial_level_baseline_vs_augmented.csv` |
| Figure 2 (*was* / *being* coefficients) | `results/08_was_being_current_plus_preceding_surprisal.csv` |
| Figure 1, Figure A1 | `figures/` |

## Stimuli

The target and filler sentences are taken verbatim from Appendix A (Experiment 1) of
Nelson, M. J., et al. (2021). Only the 32 target sentences are redistributed here; see the paper for
the full citation.

## Citation

If you use these data or scripts, please cite the paper (citation to be added on publication).
A `CITATION.cff` file is provided for GitHub's citation widget.

## License

Code: MIT (see `LICENSE`). Data (`data/*.csv`): CC BY 4.0.

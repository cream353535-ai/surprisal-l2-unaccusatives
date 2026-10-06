#!/usr/bin/env bash
# Reproduce everything from the repository root.
set -e
python scripts/compute_surprisal.py
Rscript scripts/analyze_surprisal.R
python scripts/make_figures.py

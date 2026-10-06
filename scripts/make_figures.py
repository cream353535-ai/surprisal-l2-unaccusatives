"""make_figures.py — Figure 1(a), 1(b), and Appendix Figure A1 from the project data.
Inputs: data/human_rt_trials_pos4-8.csv, data/surprisal_long.csv
Outputs: figures/Fig1_human_RT_pos4-8.pdf/.png, figures/Fig_surprisal_by_position.pdf/.png,
         figures/FigA1_surprisal_by_model.pdf/.png
Run from the repository root: python scripts/make_figures.py
"""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; DATA = ROOT / "data"; FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)

POS = ["Verb", "Det", "NP2", "was", "being"]
COND = ["AA", "IA", "AN", "IN"]
LABEL = {"AA": "Animate + Alternating", "IA": "Inanimate + Alternating",
         "AN": "Animate + Non-alternating", "IN": "Inanimate + Non-alternating"}
STYLE = {"AA": dict(color="#1f77b4", ls="-",  marker="o"),
         "IA": dict(color="#1f77b4", ls="--", marker="s"),
         "AN": dict(color="#d62728", ls="-",  marker="o"),
         "IN": dict(color="#d62728", ls="--", marker="s")}
MODELS = [("gpt2-small", "GPT-2 Small"), ("gpt2-medium", "GPT-2 Medium"), ("gpt2-large", "GPT-2 Large"),
          ("smollm2-135m", "SmolLM2 135M"), ("smollm2-360m", "SmolLM2 360M"), ("smollm2-1.7b", "SmolLM2 1.7B")]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

def plot_lines(ax, means, ses, ylabel):
    x = np.arange(len(POS))
    for c in COND:
        ax.errorbar(x, means.loc[c, POS], yerr=ses.loc[c, POS], label=LABEL[c],
                    capsize=2, lw=1.2, ms=4, **STYLE[c])
    ax.set_xticks(x); ax.set_xticklabels(POS)
    for lab in ax.get_xticklabels():
        if lab.get_text() in ("was", "being"): lab.set_fontstyle("italic")
    ax.set_ylabel(ylabel)

# ---------- Figure 1(a): human RT, SE of participant means ----------
hum = pd.read_csv(DATA / "human_rt_trials_pos4-8.csv")
hum = hum[(hum.rt >= 100) & (hum.rt <= 3000)]          # same trimming as analyze_surprisal.R
pm = hum.groupby(["subject", "condition", "position"])["rt"].mean().reset_index()
m1 = pm.groupby(["condition", "position"])["rt"].mean().unstack()
s1 = pm.groupby(["condition", "position"])["rt"].sem().unstack()
fig, ax = plt.subplots(figsize=(4.2, 3.2))
plot_lines(ax, m1, s1, "Mean reading time (ms)")
ax.set_xlabel("Word position"); ax.legend(frameon=False, fontsize=7.5)
fig.tight_layout()
for ext in ("pdf", "png"): fig.savefig(FIG / f"Fig1_human_RT_pos4-8.{ext}", dpi=300)

# ---------- Figure 1(b): surprisal, mean over 6 models, SE across models ----------
sur = pd.read_csv(DATA / "surprisal_long.csv").query("4 <= wpos <= 8")
bm = sur.groupby(["model", "condition", "position"])["surprisal"].mean().reset_index()
m2 = bm.groupby(["condition", "position"])["surprisal"].mean().unstack()
s2 = bm.groupby(["condition", "position"])["surprisal"].sem().unstack()
fig, ax = plt.subplots(figsize=(4.2, 3.2))
plot_lines(ax, m2, s2, "Mean surprisal (bits)")
ax.set_xlabel("Word position"); ax.legend(frameon=False, fontsize=7.5)
fig.tight_layout()
for ext in ("pdf", "png"): fig.savefig(FIG / f"Fig_surprisal_by_position.{ext}", dpi=300)

# ---------- Figure A1: surprisal per model ----------
fig, axes = plt.subplots(2, 3, figsize=(9, 5.6), sharey=True)
for ax, (key, name) in zip(axes.flat, MODELS):
    d = sur[sur.model == key]
    m = d.groupby(["condition", "position"])["surprisal"].mean().unstack()
    s = d.groupby(["condition", "position"])["surprisal"].sem().unstack()   # SE across the 8 items
    plot_lines(ax, m, s, "Surprisal (bits)" if ax in axes[:, 0] else "")
    ax.set_title(name, fontsize=9)
axes[0, 0].legend(frameon=False, fontsize=7)
fig.tight_layout()
for ext in ("pdf", "png"): fig.savefig(FIG / f"FigA1_surprisal_by_model.{ext}", dpi=300)
print(m1.round(0)); print(m2.round(2))

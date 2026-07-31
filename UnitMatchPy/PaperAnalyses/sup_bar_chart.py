import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats


# point this path to the folder containing the results CSV files, ideally with N set by UnitMatch for all methods for a fair comparison.
root = r"/path/to/results/N_set_by_UM"
# root = r"/path/to/results"

# Point these to whatever your results CSV files are named.
dum = pd.read_csv(os.path.join(root, "NBProb18mice_results.csv")).drop_duplicates()
um = pd.read_csv(os.path.join(root, "MatchProbNew_results.csv")).drop_duplicates()
unt = pd.read_csv(os.path.join(root, "NBProbuntrained_results.csv")).drop_duplicates()
untae = pd.read_csv(
    os.path.join(root, "NBProbuntrainedAE_results.csv")
).drop_duplicates()
unf = pd.read_csv(os.path.join(root, "NBProbunfinetuned_results.csv")).drop_duplicates()

models = ["_UM", "_DUM", "_UNTAE", "_UNF", "_UNT"]
N_cols = []
for df, name in zip([um, dum, untae, unf, unt], models):
    df.rename(columns={"AUCisi": f"AUCisi{name}"}, inplace=True)
    df.rename(columns={"N": f"N{name}"}, inplace=True)
    df.drop(columns=["AUCrpc"], inplace=True)
    N_cols.append(f"N{name}")

merge_keys = ["mouse", "probe", "loc", "r1", "r2", "day1", "day2", "delta_days"]
merged_full = dum.merge(um, on=merge_keys)
merged_full = merged_full.merge(unt, on=merge_keys)
merged_full = merged_full.merge(untae, on=merge_keys)
merged_full = merged_full.merge(unf, on=merge_keys)
merged_full.drop_duplicates(inplace=True)

# Filter out session pairs with fewer than 20 matches (AUC is noisy under this threshold)
merged_full["maxN"] = merged_full[N_cols].max(axis=1)
merged_full = merged_full.loc[merged_full["maxN"] > 19]
print(len(merged_full))

AUC_means = {}
N_sessions = {}

for mouse, group in merged_full.groupby("mouse"):
    AUC_means[mouse] = []
    N_sessions[mouse] = []
    for model in models:
        AUC_means[mouse].append(group[f"AUCisi{model}"].mean())
        N_sessions[mouse].append(len(group))
AUC_means = pd.DataFrame(AUC_means)

DeepUM = AUC_means.iloc[1]
OldUM = AUC_means.iloc[0]
pval = stats.wilcoxon(DeepUM, OldUM).pvalue
print(f"P-value comparing UnitMatch with DeepUnitMatch(Wilcoxon): {pval:.4f}")

# for each model, plot a bar graph of mean AUC across mice with error bars showing std deviation
fig, ax = plt.subplots(figsize=(10, 6))
colors = ["b", "r", "r", "r", "r", "r"]
alphas = [1, 1, 0.5, 0.5, 0.3]
fills = list(zip(colors, alphas))
x = np.arange(len(models))
means = [AUC_means.iloc[idx].mean() for idx, row in AUC_means.iterrows()]
stds = [AUC_means.iloc[idx].std() for idx, row in AUC_means.iterrows()]

# Create bars individually to apply different alpha values and solid borders
for i in range(len(x)):
    ax.bar(
        x[i],
        means[i],
        yerr=stds[i],
        capsize=5,
        color=fills[i],
        edgecolor=colors[i],
        linewidth=1,
    )

ax.set_xticks(x)
ax.set_yticks(np.arange(0.80, 0.98, 0.15))
ax.set_ylabel("Mean AUC")
ax.set_ylim(0.80, 0.98)
ax.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xticklabels(
    [
        "UnitMatch",
        "DeepUnitMatch",
        "Untrained AE",
        "Unfinetuned",
        "Untrained",
    ]
)
for i, mouse in enumerate(AUC_means.columns):
    ax.plot(x, AUC_means[mouse], marker="o", linestyle="--", alpha=0.5, c="grey")

# add mean values as text above bars
for i, (mean, std) in enumerate(zip(means, stds)):
    ax.text(i, 1.0, f"{mean:.3f}", ha="center", va="bottom")

plt.title(f"P-value (Wilcoxon): {pval:.4f}")
plt.rcParams["svg.fonttype"] = "none"
ax = plt.gca()
ax.spines[["right", "top"]].set_visible(False)
plt.show()

import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
from sklearn.neighbors import KernelDensity

# === USER CONFIG ===
# Point this at the directory containing the recording data
# (the folder that holds AL032/19011111882/... ). Edit for your machine.
ALL_DATA_DIR = "/absolute/path/to/ALL_DATA"

# Figures are written to a "figures" directory alongside this repo (a sibling
# of the repo root). Derived automatically — no need to edit.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGURES_DIR = os.path.join(os.path.dirname(REPO_ROOT), "figures")
fig_dir = os.path.join(FIGURES_DIR, "ISIhistograms")
os.makedirs(fig_dir, exist_ok=True)

spk_path1 = os.path.join(ALL_DATA_DIR, "AL032", "19011111882", "2", "1", "spikes.npy")
spk_path2 = os.path.join(ALL_DATA_DIR, "AL032", "19011111882", "2", "1_5", "spikes.npy")
with h5py.File(spk_path1, "r") as f:
    clusters1 = f["spkclus"][()]
    times1 = f["spktimes"][()]
with h5py.File(spk_path2, "r") as f:
    clusters2 = f["spkclus"][()]
    times2 = f["spktimes"][()]

unique_clus = np.unique(clusters1)

neuron1 = times1[clusters1 == 247]
neuron2 = times1[clusters1 == 0]
neuron2 = neuron2[len(neuron2) // 2 :]
match1 = times2[clusters2 == 270]
match1 = match1[len(match1) // 2 :]

half1 = neuron1[: len(neuron1) // 2]
half2 = neuron1[len(neuron1) // 2 :]
print("Half 1 spikes:", half1.shape)
print("Half 2 spikes:", half2.shape)

ISIs_half1 = np.diff(half1)
ISIs_half2 = np.diff(half2)

# use log bins for histogram
# isi histograms for two halves of same neuron
fig, ax = plt.subplots(figsize=(10, 6))
logbins = 5 * 10 ** np.arange(-4, 0.1, 0.1)
ax.hist(np.diff(half1), bins=logbins, color="black", histtype="step", density=True)

ax.hist(np.diff(half2), bins=logbins, color="grey", histtype="step", density=True)
ax.set_title("ISI Histograms - Same unit, different halves")
ax.spines[["right", "top"]].set_visible(False)
ax.spines["left"].set_position(("outward", 15))
ax.spines["bottom"].set_position(("outward", 15))
ax.set_xscale("log")
ax.set_xlabel("Inter-Spike Interval (s)")
# ax.set_yticks([])
plt.rcParams["svg.fonttype"] = "none"
plt.savefig(
    os.path.join(fig_dir, "same_unit.svg"), dpi=300, bbox_inches="tight", format="svg"
)
plt.tight_layout()
plt.show()

# # isi histograms for two different neurons within a session
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(np.diff(half1), bins=logbins, color="black", histtype="step", density=True)
ax.hist(np.diff(neuron2), bins=logbins, color="orange", histtype="step", density=True)
ax.set_title("ISI Histograms - Different units")
ax.spines[["right", "top"]].set_visible(False)
ax.spines["left"].set_position(("outward", 15))
ax.spines["bottom"].set_position(("outward", 15))
ax.set_xscale("log")
ax.set_xlabel("Inter-Spike Interval (s)")
ax.set_xticklabels([])
plt.rcParams["svg.fonttype"] = "none"
plt.savefig(
    os.path.join(fig_dir, "different_units.svg"),
    dpi=300,
    bbox_inches="tight",
    format="svg",
)
plt.tight_layout()
plt.show()


# isi histograms for matched units across sessions
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(np.diff(half1), bins=logbins, color="black", histtype="step", density=True)
ax.hist(np.diff(match1), bins=logbins, color="green", histtype="step", density=True)
ax.set_title("ISI Histograms - Matched units across sessions")
ax.spines[["right", "top"]].set_visible(False)
ax.spines["left"].set_position(("outward", 15))
ax.spines["bottom"].set_position(("outward", 15))
ax.set_xscale("log")
ax.set_xlabel("Inter-Spike Interval (s)")
ax.set_xticklabels([])
plt.rcParams["svg.fonttype"] = "none"
plt.savefig(
    os.path.join(fig_dir, "matched_units.svg"),
    dpi=300,
    bbox_inches="tight",
    format="svg",
)
plt.tight_layout()
plt.show()

# # find match
# conn = sqlite3.connect('matchtables.db')
# mt = pd.read_sql_query("SELECT ID1,ID2,RecSes1,RecSes2,NBProb18mice,newISI FROM AL032_19011111882_2", conn)
# df = pick(mt, 4, 5)
# matches = get_matches_1model(df, "NBProb18mice")

# match_isi = mt.loc[matches, 'newISI'].values
# same_isi = df.loc[(df['RecSes1'] == df['RecSes2']) & (df['ID1']==df['ID2']), 'newISI'].values
# diff_isi = df.loc[(df['RecSes1'] == df['RecSes2']) & (df['ID1']!=df['ID2']), 'newISI'].values

# # save these values
# np.savez(os.path.join(fig_dir, "ISI_values.npz"), same_isi=same_isi, diff_isi=diff_isi, match_isi=match_isi)

# load these values
npz_path = os.path.join(fig_dir, "ISI_values.npz")
if os.path.exists(npz_path):
    data = np.load(npz_path)
    same_isi = data["same_isi"]
    diff_isi = data["diff_isi"]
    match_isi = data["match_isi"]

    # plot histograms of ISI distances
    fig, ax = plt.subplots(figsize=(10, 6))

    # plot KDEs
    kde_same = KernelDensity(kernel="gaussian", bandwidth=0.01).fit(
        same_isi.reshape(-1, 1)
    )
    kde_diff = KernelDensity(kernel="gaussian", bandwidth=0.01).fit(
        diff_isi.reshape(-1, 1)
    )
    kde_match = KernelDensity(kernel="gaussian", bandwidth=0.01).fit(
        match_isi.reshape(-1, 1)
    )
    x = np.linspace(0.1, 0.99, 1000).reshape(-1, 1)
    ax.plot(x, np.exp(kde_same.score_samples(x)), color="grey")
    ax.plot(x, np.exp(kde_diff.score_samples(x)), color="orange")
    ax.plot(x, np.exp(kde_match.score_samples(x)), color="green")

    ax.set_xlabel("ISI histogram correlation")
    ax.spines[["right", "top"]].set_visible(False)
    ax.spines["left"].set_position(("outward", 15))
    ax.spines["bottom"].set_position(("outward", 15))
    plt.rcParams["svg.fonttype"] = "none"
    plt.savefig(
        os.path.join(fig_dir, "ISI_histogram_correlations.svg"),
        dpi=300,
        bbox_inches="tight",
        format="svg",
    )
    plt.show()
else:
    print(f"Skipping ISI-correlation KDE plot: {npz_path} not found")

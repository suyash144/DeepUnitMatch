import numpy as np
import h5py
import matplotlib.pyplot as plt
import os

# === USER CONFIG ===
# Point this at the directory containing the recording data
# (the folder that holds AL032/19011111882/... ). Edit for your machine.
ALL_DATA_DIR = "/path/to/ALL_DATA"  # Update this path to your local data directory

# Figures are written to a "figures" directory alongside this repo (a sibling
# of the repo root). Derived automatically — no need to edit.
REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
FIGURES_DIR = os.path.join(os.path.dirname(REPO_ROOT), "figures")
fig_dir = os.path.join(FIGURES_DIR, "ISIhistograms")
os.makedirs(fig_dir, exist_ok=True)

neuron_path = os.path.join(ALL_DATA_DIR, "AL032", "19011111882", "2", "1", "processed_waveforms", "Unit247_RawSpikes.npy")
match_path = os.path.join(ALL_DATA_DIR, "AL032", "19011111882", "2", "1_5", "processed_waveforms", "Unit270_RawSpikes.npy")
nonmatch_path = os.path.join(ALL_DATA_DIR, "AL032", "19011111882", "2", "1_5", "processed_waveforms", "Unit2_RawSpikes.npy")

with h5py.File(neuron_path, "r") as f:
    neuron = f["waveform"][()]
    neuron = neuron.mean(axis=-1)  # previously split into mean of two halves; not needed here
    neuron = neuron[:, 15]  # take middle channel for plotting
with h5py.File(match_path, "r") as f:
    match = f["waveform"][()]
    match = match.mean(axis=-1)
    match = match[:, 15]
with h5py.File(nonmatch_path, "r") as f:
    non_match = f["waveform"][()]
    non_match = non_match.mean(axis=-1)
    non_match = non_match[:, 15]

# Plot waveforms of matched units across sessions
plt.plot(neuron, label="Neuron 247 (session 1)")
plt.plot(match, label="Match found by DeepUnitMatch")
plt.plot(non_match, label="Non-matching unit")

plt.xlabel("Time (ms)")
plt.ylabel("Amplitude (a.u.)")
plt.legend()
plt.show()



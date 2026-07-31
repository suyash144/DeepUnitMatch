import numpy as np
import pandas as pd
import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from testing.test import load_trained_model
from utils.helpers import get_unit_id
from utils.npdataset import NeuropixelsDataset_cortexlab, ValidationExperimentBatchSampler


def embed_data(
    data_dir,
    ks_dirs=None,
    unit_order: str = "filesystem",
    load_bombcell: bool = False,
):
    """
    Embed every unit in `data_dir` using the trained DeepUnitMatch model.

    Required data structures
    -------------------------
    data_dir : a directory whose immediate children are per-session subfolders
        (named by session index, e.g. "0", "1", ...), each containing
        `Unit{id}_RawSpikes.npy` HDF5 files with `waveform` ([T, C, 2]) and
        `MaxSitepos` ([2]) datasets. This is exactly the `processed_waveforms`
        folder produced by `param_fun.get_snippets(...)` in the main
        `DeepUnitMatch.ipynb` demo notebook, and the same directory passed to
        `testing.test.inference`.

    ks_dirs : only required when `load_bombcell=True`. A list of raw
        KiloSort/UnitMatch output directory paths, one per session, in the
        *same order* the sessions were originally passed to `get_snippets`
        (i.e. the same list as `KS_dirs` in the main demo notebook,
        index-aligned with the session folders in `data_dir`). Each entry is
        expected to have a `qMetrics/templates._bc_qMetrics.parquet` file
        (Bombcell's output, historically fetched from the cortexlab server
        share) -- see `get_bombcell_data` for the expected column schema.

    Returns
    -------
    A dict with one row per unit:
        "embedded_first" / "embedded_second": (N, 256) DNN embeddings for
            each cross-validation half.
        "session_id": int session index (position of that unit's session
            folder in `sorted(os.listdir(data_dir))`).
        "unit_id": the unit's cluster ID within its session.
        "depth": the unit's depth (MaxSitepos[:, 1]).
        "bombcell": (only when `load_bombcell=True`) dict of Bombcell QC
            columns -> list of values, aligned with the other arrays.
    """
    if load_bombcell and not ks_dirs:
        raise ValueError("ks_dirs is required when load_bombcell=True")

    model = load_trained_model()
    test_dataset = NeuropixelsDataset_cortexlab(data_dir, unit_order=unit_order)
    test_sampler = ValidationExperimentBatchSampler(test_dataset, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler)

    embedded_first, embedded_second = [], []
    session_ids, unit_ids_all, depths = [], [], []
    bc_values = {} if load_bombcell else None

    with torch.no_grad():
        for waveforms_fh, waveforms_sh, msp, exp_ids_i, filepaths_i in tqdm(test_loader):
            bsz_i = waveforms_fh.shape[0]
            session_id = int(exp_ids_i[0])  # same for every file in the batch
            unit_ids = [get_unit_id(f) for f in filepaths_i]

            enc_fh = model(waveforms_fh).numpy()  # shape [bsz, 256]
            enc_sh = model(waveforms_sh).numpy()  # shape [bsz, 256]
            embedded_first.extend(enc_fh)
            embedded_second.extend(enc_sh)
            session_ids.extend([session_id] * bsz_i)
            unit_ids_all.extend(unit_ids)
            depths.extend(msp[:, 1].tolist())

            if load_bombcell:
                n_before = len(bc_values.get("clusterID", []))
                bc_values, _ = get_bombcell_data(
                    bc_values, ks_dirs[session_id], unit_ids=unit_ids
                )
                n_added = len(bc_values["clusterID"]) - n_before
                if n_added != bsz_i:
                    raise ValueError(
                        f"Session {session_id}: expected {bsz_i} Bombcell rows for "
                        f"this batch's units, found {n_added}. Check that ks_dirs is "
                        f"index-aligned with the session folders in data_dir."
                    )

    result = {
        "embedded_first": np.array(embedded_first),
        "embedded_second": np.array(embedded_second),
        "session_id": np.array(session_ids),
        "unit_id": np.array(unit_ids_all),
        "depth": np.array(depths),
    }
    if load_bombcell:
        result["bombcell"] = bc_values
    return result


def get_bombcell_data(existing, exp_server_path, unit_ids=None):
    """
    Load Bombcell data for one session.

    Called once per session by `embed_data` (one session == one batch, since
    `ValidationExperimentBatchSampler` batches by experiment).

    Args:
        - existing: dictionary of bombcell parameters -> list of values for each neuron.
        - exp_server_path: this session's raw KiloSort/UnitMatch output directory (an
          entry of `ks_dirs`), expected to contain
          `qMetrics/templates._bc_qMetrics.parquet` with (at least) the columns listed
          below.
        - unit_ids: the IDs to get data for (only want good IDs, and post-merge)
    """
    parquet_path = os.path.join(
        exp_server_path, "qMetrics", "templates._bc_qMetrics.parquet"
    )
    columns = [
        "clusterID",
        "percentageSpikesMissing_gaussian",
        "presenceRatio",
        "nSpikes",
        "nPeaks",
        "nTroughs",
        "isSomatic",
        "waveformDuration_peakTrough",
        "spatialDecaySlope",
        "waveformBaselineFlatness",
        "rawAmplitude",
    ]
    bc = pd.read_parquet(parquet_path)
    bc["clusterID"] = (bc["clusterID"] - 1).astype(int)

    if unit_ids is not None:
        bc = bc.loc[bc["clusterID"].isin(unit_ids)]

    if len(existing.keys()) == 0:
        for col in columns:
            existing[col] = bc[col].values.tolist()
    else:
        for col in columns:
            if col not in bc.columns:
                existing[col].extend([np.nan] * len(bc))
            else:
                existing[col].extend(bc[col].values.tolist())

    return existing, False

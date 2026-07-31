import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import mat73
import sqlite3

from UnitMatchPy.DeepUnitMatch.utils.helpers import PROJECT_ROOT


def create_sim_mat(df, col):
    sessions = df["RecSes1"].iloc[1] == df["RecSes2"].iloc[1]
    if sessions:
        l1 = l2 = int(np.sqrt(len(df)))
    else:
        l1 = len(df["ID1"].unique())
        l2 = len(df["ID2"].unique())
        assert l1 * l2 == len(df)
    if col == "index":
        vals = np.array(df.index).astype(int)
    else:
        vals = np.array(df[col])
    vals = vals.reshape((l1, l2))
    return vals


def create_concat_mat(
    df11, df12, df21, df22, col, sort_method="id", rec1=None, rec2=None, depths=None
):
    s11 = create_sim_mat(df11, col)
    s12 = create_sim_mat(df12, col)
    s21 = create_sim_mat(df21, col)
    s22 = create_sim_mat(df22, col)

    if sort_method == "depth":
        s11 = reorder_by_depth(s11, depths, rec1, rec1)
        s12 = reorder_by_depth(s12, depths, rec1, rec2)
        s21 = reorder_by_depth(s21, depths, rec2, rec1)
        s22 = reorder_by_depth(s22, depths, rec2, rec2)

    top_row = np.concatenate((s11, s12), axis=1)
    bottom_row = np.concatenate((s21, s22), axis=1)

    sim_matrix = np.concatenate((top_row, bottom_row))
    return sim_matrix


def read_depths(mouse, probe, loc):
    base = r"\\znas\Lab\Share\UNITMATCHTABLES_ENNY_CELIAN_JULIE\FullAnimal_KSChanMap"
    # Find Unitmatch.mat for each recording
    um_path = os.path.join(base, mouse, probe, loc, "UnitMatch", "UnitMatch.mat")
    um = mat73.loadmat(um_path, verbose=False)
    pl = um["WaveformInfo"]["ProjectedLocation"]  # shape [3 x Nclus x 2]
    x = pl[1, :, :]  # shape [Nclus x 2]
    y = pl[2, :, :]
    x = np.array(np.round(np.mean(x, axis=1), decimals=-2))
    y = np.array(np.mean(y, axis=1))
    if len(np.unique(x)) != 4:
        print(f"CAUTION: Found {len(np.unique(x))} clusters of x values rather than 4.")
    uid = um["UniqueIDConversion"]["OriginalClusID"]
    urs = um["UniqueIDConversion"]["recsesAll"]
    goodids = um["UniqueIDConversion"]["GoodID"]
    uid, urs = uid[goodids == 1], urs[goodids == 1]
    depth = x * 1e6 + y
    depth_dict = {"RecSes": urs, "ID": uid, "IDrank": "", "depth": depth}
    depth_df = pd.DataFrame(depth_dict)
    return depth_df


def compare_two_recordings(
    df: pd.DataFrame, rec1: int, rec2: int, sort_method="id", depths=None, vis=False
):
    """
    df: matchtable as a pandas DataFrame
    rec1: integer corresponding to the RecSes1 that you want to select
    rec2: integer corresponding to the RecSes2 that you want to select
    sort_method: how you want the results to be sorted (depth or id)
    depths: only required if sort_method="depth". can read depths directly from matlab using the
    read_depths function.
    """
    # Pick out the relevant columns and ensure they are sorted
    col = df.loc[:, "WavformSim":"LocTrajectorySim"]
    df["NoLocScore"] = col.mean(axis=1)
    df = df.loc[
        :, ["RecSes1", "RecSes2", "ID1", "ID2", "DNNSim", "MatchProb", "NoLocScore"]
    ]
    df11 = df.loc[(df["RecSes1"] == rec1) & (df["RecSes2"] == rec1), :]
    df12 = df.loc[(df["RecSes1"] == rec1) & (df["RecSes2"] == rec2), :]
    df21 = df.loc[(df["RecSes1"] == rec2) & (df["RecSes2"] == rec1), :]
    df22 = df.loc[(df["RecSes1"] == rec2) & (df["RecSes2"] == rec2), :]

    if sort_method == "depth":
        sim_matrix = create_concat_mat(
            df11, df12, df21, df22, "DNNSim", "depth", rec1, rec2, depths
        )
        indices = create_concat_mat(
            df11, df12, df21, df22, "index", "depth", rec1, rec2, depths
        )
        um_output = create_concat_mat(
            df11, df12, df21, df22, "MatchProb", "depth", rec1, rec2, depths
        )
        um_score = create_concat_mat(
            df11, df12, df21, df22, "NoLocScore", "depth", rec1, rec2, depths
        )
    elif sort_method == "id":
        sim_matrix = create_concat_mat(df11, df12, df21, df22, "DNNSim", "id")
        um_output = create_concat_mat(df11, df12, df21, df22, "MatchProb")
        um_score = create_concat_mat(df11, df12, df21, df22, "NoLocScore")
    else:
        raise ValueError("""Please pick a sorting method from 'depth' or 'id'
                         Default is id as this requires no info about spatial positions of neurons.
                         Depth gives better results though.""")
    if vis:
        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
        ax1.matshow(sim_matrix)
        ax1.set_title("DNN Similarity matrix")

        ax2.matshow(um_output)
        ax2.set_title("UnitMatch match probabilities")

        ax3.matshow(um_score)
        ax3.set_title("UnitMatch score (no centroid)")

        plt.show()
    return sim_matrix, indices


def get_cross_day_matrices(df: pd.DataFrame, rec1: int, rec2: int, depths=None):

    df11 = df.loc[(df["RecSes1"] == rec1) & (df["RecSes2"] == rec1), :]
    df12 = df.loc[(df["RecSes1"] == rec1) & (df["RecSes2"] == rec2), :]
    df21 = df.loc[(df["RecSes1"] == rec2) & (df["RecSes2"] == rec1), :]
    df22 = df.loc[(df["RecSes1"] == rec2) & (df["RecSes2"] == rec2), :]
    s12 = reorder_by_depth(create_sim_mat(df12, "DNNSim"), depths, rec1, rec2)
    s21 = reorder_by_depth(create_sim_mat(df21, "DNNSim"), depths, rec2, rec1)
    indices = create_concat_mat(
        df11, df12, df21, df22, "index", "depth", rec1, rec2, depths
    )

    return s12, s21, indices


def reorder_by_depth(matrix: np.ndarray, depths, recses1: int, recses2: int):
    """
    Reorders a square block matrix by depth within each session's block.

    This function assumes the matrix has a shape of (n1+n2) x (n1+n2), where
    the first n1 rows/columns correspond to neurons from `recses1` and the
    following n2 rows/columns correspond to neurons from `recses2`.

    The reordering sorts the first n1 rows/cols by the depth of recses1 neurons,
    and sorts the next n2 rows/cols by the depth of recses2 neurons,
    preserving the overall block structure.

    Args:
        matrix (np.ndarray): The square comparison matrix, shape (n1+n2, n1+n2).
        depths (pd.DataFrame): DataFrame with "RecSes" and "depth" columns.
        recses1 (int): The ID for the first recording session (first block).
        recses2 (int): The ID for the second recording session (second block).

    Returns:
        np.ndarray: The reordered matrix.
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square.")

    # 1. Filter depths for each session and get their counts.
    depths1 = depths[depths["RecSes"] == recses1].copy()
    depths2 = depths[depths["RecSes"] == recses2].copy()
    n1, n2 = len(depths1), len(depths2)

    # 2. Validate that the total number of neurons matches the matrix dimension.
    if n1 + n2 != matrix.shape[0]:
        raise ValueError(
            f"The total number of neurons from recses {recses1} ({n1}) and {recses2} ({n2}) "
            f"does not match the matrix dimension ({matrix.shape[0]})."
        )

    # 3. Get the sorting indices for the first session's neurons.
    # These indices will range from 0 to n1-1, which is what we want for the first block.
    sort_indices_s1 = np.argsort(depths1["depth"].values)

    # 4. Get the sorting indices for the second session's neurons.
    # These indices also range from 0 to n2-1 initially.
    sort_indices_s2_raw = np.argsort(depths2["depth"].values)
    # We must offset them to correctly target the second block of the matrix (indices n1 to n1+n2-1).
    sort_indices_s2_offset = sort_indices_s2_raw + n1

    # 5. Concatenate the indices to create a single permutation vector for the entire matrix.
    # This vector first reorders the 0..n1-1 block, then the n1..n1+n2-1 block.
    full_sort_indices = np.concatenate([sort_indices_s1, sort_indices_s2_offset])

    # 6. Apply this single permutation vector to both the rows and columns.
    # Using np.ix_ is the canonical way to reorder rows and columns independently.
    # This correctly reorders each block while preserving its position.
    reordered_matrix = matrix[np.ix_(full_sort_indices, full_sort_indices)]

    return reordered_matrix


if __name__ == "__main__":
    conn = sqlite3.connect(os.path.join(PROJECT_ROOT, "matchtables.db"))
    df = pd.read_sql_query("SELECT * FROM AL036_19011116882_3", conn)

    depths = read_depths("AL036", "19011116882", "3")

    sim_matrix, idx = compare_two_recordings(df, 19, 20, "depth", depths, vis=True)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 13:26:30 2024

@author: samy
"""

# %%
import numpy as np
import time

# from functions_analysis import *
from pathlib import Path

from fun_loaddata import *
from fun_dfcspeed import *

from fun_metaconnectivity import (
    compute_mc_nplets_mask_and_index,
    compute_metaconnectivity,
    intramodule_indices_mask,
    get_fc_mc_indices,
    get_mc_region_identities,
    fun_allegiance_communities,
    compute_trimers_identity,
    build_trimer_mask,
    trimers_leaves_fc,
    trimers_root_fc,
    compute_mc_nplets_mask_and_index,
)

from fun_utils import (
    set_figure_params,
    get_paths,
    load_cognitive_data,
    load_timeseries_data,
    load_grouping_data,
)

# =============================================================================
# This code compute
# Load the dataz
# Intersect the 2 and 4 months to have data that have the two datapoints
# ========================== Figure parameters ================================
save_fig = set_figure_params(False)

# =================== Paths and folders =======================================
timeseries_folder = "Timecourses_updated_03052024"
external_disk = True
if external_disk == True:
    root = Path("/media/samy/Elements1/Proyectos/LauraHarsan/script_mc/")
else:
    root = Path("/home/samy/Bureau/Proyect/LauraHarsan/Ines/")

paths = get_paths(
    external_disk=True, external_path=root, timecourse_folder=timeseries_folder
)

# ========================== Load data =========================
cog_data_filtered = load_cognitive_data(paths["sorted"] / "cog_data_sorted_2m4m.csv")
mask_groups, label_variables = load_grouping_data(
    paths["results"] / "grouping_data_oip.pkl"
)

data_ts = load_timeseries_data(paths["sorted"] / "ts_and_meta_2m4m.npz")
ts = data_ts["ts"]
n_animals = data_ts["n_animals"]
regions = data_ts["regions"]
anat_labels = data_ts["anat_labels"]
# %%
# ======================== Metaconnectivity ==========================================
# Parameters speed
PROCESSORS = -1

lag = 1
tau = 5
window_size = 7
window_parameter = (5, 100, 1)

# Parameters allegiance analysis
n_runs_allegiance = 1000
gamma_pt_allegiance = 100

tau_array = np.append(np.arange(0, tau), tau)
lentau = len(tau_array)

time_window_min, time_window_max, time_window_step = window_parameter
time_window_range = np.arange(time_window_min, time_window_max + 1, time_window_step)


# %%Reference metaconnectivity for allegiance matrix
# # ========================Set Reference parameters ==========================================

label_ref = label_variables[1][0]  # The label of the reference matrix
ind_ref = mask_groups[1][0]  # the mask of the reference matrix
# %% 
# =========================Load metaconnectivity modularity =========================
# Load the metaconnectivity modularity dataset
save_filename = paths[
    "mc_mod"
] / f"mc_allegiance_ref(runs={label_ref}_gammaval={n_runs_allegiance})={gamma_pt_allegiance}_lag={lag}_windowsize={window_size}_animals={n_animals}_regions={regions}.npz".replace(
    " ", ""
)

data_analysis = np.load(save_filename, allow_pickle=True)
mc_ref_allegiance_sort = data_analysis["mc_ref_allegiance_sort"]
mc_modules_mask = data_analysis["mc_modules_mask"]
mc_idx = data_analysis["mc_idx_tril"]
fc_idx = data_analysis["fc_idx_tril"]
mc_val = data_analysis["mc_val_tril"]
mc_mod_idx = data_analysis["mc_mod_idx"]
mc_reg_idx = data_analysis["mc_reg_idx"]
fc_reg_idx = data_analysis["fc_reg_idx"]


# %%
# ========================Trimers==========================================
# Compute trimers
# trimer_index, trimer_reg_id, trimer_apex = compute_trimers_identity(regions, allegiance_sort=mc_ref_allegiance_sort)

# # Build trimer mask
# n_fc_edges = int(regions * (regions - 1) / 2)
# mc_nplets_mask = build_trimer_mask(trimer_index, trimer_apex, n_fc_edges)
# mc_nplets_index = mc_nplets_mask[mc_idx[:, 0], mc_idx[:, 1]]
# tril_i, tril_j = np.tril_indices(n_fc_edges, k=-1)
# mc_nplets_index3 = mc_nplets_mask[tril_i, tril_j]

# print(f"Trimer processing time: {stop - start:.3f} seconds")
# %% Compute the mask and index
# ========================Trimers==========================================
# Compute the mask and index
mc_nplets_mask2, mc_nplets_index2 = compute_mc_nplets_mask_and_index(
    regions, allegiance_sort=mc_ref_allegiance_sort
)
#%%
# # ===============================================
# # Genuine trimers MC_{ir,jr}>FC_{ij}
# #Threshold for the FC_{ij}
# # =============================================================================

# Compute FC
# # animal=0
fc = np.array(
    [
        ts2fc(ts[animal], format_data="2D", method="pearson")
        for animal in range(n_animals)
    ]
)

fc_values = fc[:, fc_idx[:, 0], fc_idx[:, 1]]
fc_values_median = np.median(fc_values, axis=0)

trimers_idx = fc_reg_idx[mc_nplets_index > 0]
fc_trimers_leaves_bool = np.all(
    (fc_reg_idx * (mc_nplets_index > 0)[:, None, None]) > 0, axis=(1, 2)
)
# mc_trimers_leaves_bool = np.alltrue( (mc_reg_idx.T * (mc_nplets_index>0)[:,None]),axis=(1))



# %%
# =============================================================================
# For MC_{ir,jr} > FC_{i,j}
# =============================================================================
fc_trimers_leaves_idx = np.array(
    [trimers_leaves_fc(tri_idx) for tri_idx in trimers_idx]
)  # trimers leaves region number
fc_leaves_values = fc[
    :, fc_trimers_leaves_idx[:, 0] - 1, fc_trimers_leaves_idx[:, 1] - 1
]  # trimer leaves values
trimers_genuine_mc_root_fc_leaves = (mc_val[:, (mc_nplets_index > 0)]) > (
    fc_leaves_values
)  # genuine trimers by MC_{ir,jr} > FC_{i,j}

# %%
# =============================================================================
# For FC_{ir} > FC_{i,j} or FC_{jr} > FC_{i,j}
# =============================================================================
fc_trimers_root_idx = np.squeeze([trimers_root_fc(tri_idx) for tri_idx in trimers_idx])
fc_root_values1 = fc[:, fc_trimers_root_idx - 1, fc_trimers_leaves_idx[:, 0] - 1]
fc_root_values2 = fc[:, fc_trimers_root_idx - 1, fc_trimers_leaves_idx[:, 1] - 1]
fc_root_min = np.minimum(np.abs(fc_root_values1), np.abs(fc_root_values2))

trimers_genuine_fc_root_leaves = (fc_root_min) > (fc_leaves_values)


# %%
# =============================================================================
# For MC_{ir,jr} > dFC_{i,j} and given time windows
# =============================================================================
def ts2dfc_stream(ts, windows_size, lag=None, format_data="2D", method="pearson"):
    """
    Calculate dynamic functional connectivity stream (dfc_stream) from time series data.

    Parameters:
    ts (array): Time series data of shape (t, n), where t is timepoints, n is regions.
    windows_size (int): Window size to slide over the ts.
    lag (int): Shift value for the window. Defaults to W if not specified.
    format (str): Output format. '2D' for a (l, F) shape, '3D' for a (n, n, F) shape.

    Returns:
    dFCstream (array): Dynamic functional connectivity stream.
    """

    t_total, n = np.shape(ts)
    # Not overlap
    if lag is None:
        lag = windows_size

    n_pairs = n * (n - 1) // 2  # number of pairwise correlations
    # Calculate the number of frames/windows
    frames = (t_total - windows_size) // lag + 1

    if format_data == "2D":
        dfc_stream = np.empty((n_pairs, frames))
    elif format_data == "3D":
        dfc_stream = np.empty((n, n, frames))

    for k in range(frames):
        wstart = k * lag
        wstop = wstart + windows_size
        if format_data == "2D":
            dfc_stream[:, k] = ts2fc(
                ts[wstart:wstop, :], "1D", method=method
            )  # Assuming TS2FC returns a vector
        elif format_data == "3D":
            dfc_stream[:, :, k] = ts2fc(
                ts[wstart:wstop, :], "2D", method=method
            )  # Assuming TS2FC returns a matrix

    return dfc_stream


dfc_stream = np.array(
    [
        ts2dfc_stream(
            ts[animal], window_size, lag=lag, format_data="3D", method="pearson"
        )
        for animal in range(n_animals)
    ]
)

dfc_leaves_values = dfc_stream[
    :, fc_trimers_leaves_idx[:, 0] - 1, fc_trimers_leaves_idx[:, 1] - 1
]
dfc_leaves_values_mean = np.mean(dfc_leaves_values, axis=-1)
# trimers_leaves_fc(dfc_stream)
# %%
trimers_genuine_mc_root_dfc_leaves = (mc_val[:, (mc_nplets_index > 0)]) > (
    dfc_leaves_values_mean
)


# %%

from matplotlib import pyplot as plt

label_fc_root_fc_leaves = r"$min(FC_{i,r}, FC_{j,r}) > FC_{i,j}$"
label_mc_root_fc_leaves = r"$MC_{ir,jr} > FC_{i,j}$"
label_mc_root_dfc_leaves = r"$MC_{ir,jr} > mean(dFC_{i,j})$"


plt.figure(1)
plt.clf()
plt.subplot(311)
plt.scatter(
    np.sum(trimers_genuine_fc_root_leaves, axis=0) / n_animals,
    np.sum(trimers_genuine_mc_root_fc_leaves, axis=0) / n_animals,
    alpha=0.4,
    s=3,
    # label =label_fc_root_fc_leaves + ' vs ' + label_mc_root_fc_leaves
)

plt.plot([0, 1], [0, 1], color="red", linestyle="--", linewidth=1)
plt.xlabel(label_fc_root_fc_leaves)
plt.ylabel(label_mc_root_fc_leaves)


plt.subplot(312)
plt.scatter(
    np.sum(trimers_genuine_fc_root_leaves, axis=0) / n_animals,
    np.sum(trimers_genuine_mc_root_dfc_leaves, axis=0) / n_animals,
    alpha=0.4,
    s=3,
    c="C1",
    # label =label_fc_root_fc_leaves + ' vs ' + label_mc_root_fc_leaves
)

plt.plot([0, 1], [0, 1], color="red", linestyle="--", linewidth=1)
plt.xlabel(label_fc_root_fc_leaves)
plt.ylabel(label_mc_root_dfc_leaves)

plt.subplot(313)
plt.scatter(
    np.sum(trimers_genuine_mc_root_fc_leaves, axis=0) / n_animals,
    np.sum(trimers_genuine_mc_root_dfc_leaves, axis=0) / n_animals,
    alpha=0.4,
    s=3,
    c="C2",
    # label =label_fc_root_fc_leaves + ' vs ' + label_mc_root_fc_leaves
)

plt.plot([0, 1], [0, 1], color="red", linestyle="--", linewidth=1)
plt.xlabel(label_mc_root_fc_leaves)
plt.ylabel(label_mc_root_dfc_leaves)
plt.tight_layout()
# , markersize=1)
# plt.subplot(311)
# plt.plot(np.sum(trimers_genuine_fc_root_leaves, axis=0),'.')
# plt.subplot(312)
# plt.plot(np.sum(trimers_genuine_mc_root_fc_leaves, axis=0),'.')
# plt.subplot(313)
# plt.plot(np.sum(trimers_genuine_mc_root_dfc_leaves, axis=0),'.')
# plt.imshow(fc[:,fc_idx[:,0],fc_idx[:,1]].T,
#            interpolation='none',
#            aspect='auto',
#            cmap = 'coolwarm',
#            )
# plt.colorbar()
# plt.clim(-0.6,0.6)
# %%

plt.figure(2, figsize=(12, 8))
plt.clf()
offset = 0.07  # vertical offset between time series
# for i, ts1 in enumerate(ts[0].T):
# plt.plot(ts1 + i * offset, label=f"TS {i+1}")
# plt.ylim(-0.1,0.75)
plt.title("MC(i,j)")
plt.ylabel(r"$MC_{(ij, (kl)^{N2 (N2-1)/2)})}$")
plt.xlabel("Time")
plt.tight_layout()
plt.show()
# plt.savefig(paths['figures'] / 'trimers_genuine.png', dpi=300, bbox_inches='tight')


# %%

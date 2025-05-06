#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merged script for computing genuine trimers, metaconnectivity, and modularity.

Combines functionality from:
- compute_metaconnectivity_modularity.py
- compute_metaconnectivity.py
- compute_genuine_trimers.py

@author: samy
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
from pathlib import Path
import pickle

from fun_loaddata import *
from fun_dfcspeed import ts2fc
from fun_metaconnectivity import (
    compute_metaconnectivity, 
    fun_allegiance_communities, 
    intramodule_indices_mask, 
    get_fc_mc_indices, 
    get_mc_region_identities, 
    compute_trimers_identity, 
    build_trimer_mask
)
from fun_utils import (
    set_figure_params, 
    get_paths, 
    load_cognitive_data, 
    load_timeseries_data, 
    load_grouping_data
)

# =============================================================================
# Parameters
# =============================================================================
save_fig = set_figure_params(False)

# Paths and folders
timeseries_folder = 'Timecourses_updated_03052024'
external_disk = True
paths = get_paths(external_disk=external_disk, timecourse_folder=timeseries_folder)

# Analysis parameters
PROCESSORS = -1
lag = 1
tau = 5
window_size = 7
n_runs_allegiance = 1000
gamma_pt_allegiance = 100

# =============================================================================
# Load Data
# =============================================================================
cog_data_filtered = load_cognitive_data(paths['sorted'] / 'cog_data_sorted_2m4m.csv')
data_ts = load_timeseries_data(paths['sorted'] / 'ts_and_meta_2m4m.npz')
mask_groups, label_variables = load_grouping_data(paths['results'] / "grouping_data_oip.pkl")

ts = data_ts['ts']
n_animals = data_ts['n_animals']
regions = data_ts['regions']
anat_labels = data_ts['anat_labels']
#%%
# # =============================================================================
# # Load Metaconnectivity and Modularity - if already computed
# # =============================================================================
# start = time.time()
# mc = compute_metaconnectivity(
#     ts, 
#     window_size=window_size, 
#     lag=lag, 
#     n_jobs=PROCESSORS, 
#     save_path=paths['mc']
# )
# stop = time.time()
# print(f"Metaconnectivity computation time: {stop - start:.3f} seconds")

# # =============================================================================
# # Modularity Analysis
# # =============================================================================
# label_ref = label_variables[0][0]  # Reference label
# ind_ref = mask_groups[0][0]        # Reference mask
# mc_ref = np.mean(mc[ind_ref], axis=0)

# # Compute allegiance matrix and communities
# mc_ref_allegiance_communities, mc_ref_allegiance_sort, contingency_matrix = fun_allegiance_communities(
#     mc_ref, 
#     n_runs=n_runs_allegiance, 
#     gamma_pt=gamma_pt_allegiance, 
#     ref_name=label_ref, 
#     save_path=paths['allegiance'], 
#     n_jobs=PROCESSORS
# )


# # Sort metaconnectivity by communities
# mc_allegiance = mc[:, mc_ref_allegiance_sort][:, :, mc_ref_allegiance_sort]

# # Build indices
# fc_idx, mc_idx = get_fc_mc_indices(regions)
# mc_idx = mc_idx[mc_ref_allegiance_sort]
# mc_reg_idx, fc_reg_idx = get_mc_region_identities(fc_idx, mc_idx, mc_ref_allegiance_sort)
# mc_val = mc_allegiance[:, mc_idx[:, 0], mc_idx[:, 1]]
#%% 
# =============================================================================
# Load Metaconnectivity and Modularity - if already computed
# =============================================================================
label_ref = label_variables[0][0]  # Reference label
ind_ref = mask_groups[0][0]        # Reference mask

mc_data_filename = f"mc_allegiance_ref(runs={label_ref}_gammaval={n_runs_allegiance})={gamma_pt_allegiance}_lag={lag}_windowsize={window_size}_animals={n_animals}_regions={regions}.npz".replace(' ','')

data_mc_mod_filename = paths['mc_mod'] / mc_data_filename
data_mc_mod = np.load(data_mc_mod_filename, allow_pickle=True)

#MC sorted
mc_allegiance = data_mc_mod['mc']
mc_val                 = data_mc_mod['mc_val_tril']
#mc_modules_mask                 = data_mc_mod['mc_modules_mask']
mc_idx = data_mc_mod['mc_idx_tril']
fc_idx = data_mc_mod['fc_idx_tril']

#Community values
mc_ref_allegiance_communities           = data_mc_mod['mc_ref_allegiance_communities']
mc_ref_allegiance_sort   = data_mc_mod['mc_ref_allegiance_sort']

#Indices MC, FC, regions
fc_reg_idx             = data_mc_mod['fc_reg_idx']
mc_reg_idx             = data_mc_mod['mc_reg_idx']
mc_mod_idx             = data_mc_mod['mc_mod_idx']
#%% ========================Trimers==========================================

# =============================================================================
# Compute Trimers
# =============================================================================
trimer_index, trimer_reg_id, trimer_apex = compute_trimers_identity(regions)

# Build trimer mask
n_fc_edges = int(regions * (regions - 1) / 2)
mc_nplets_mask = build_trimer_mask(trimer_index, trimer_apex, n_fc_edges)
mc_nplets_mask = mc_nplets_mask[mc_ref_allegiance_sort][:, mc_ref_allegiance_sort]
mc_nplets_index = mc_nplets_mask[mc_idx[:, 0], mc_idx[:, 1]]
#%% Save trimers
# Save the computed data
save_filename = (
    paths['trimers'] / 
    f"trimers_allegiance_ref(runs={label_ref}_gammaval={n_runs_allegiance})={gamma_pt_allegiance}_lag={lag}_windowsize={window_size}_animals={n_animals}_regions={regions}.npz".replace(' ','')
)

# Ensure the directory exists before saving
save_filename.parent.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    save_filename,
    nplets_index                      = mc_nplets_index,
    nplets_mask                      = mc_nplets_mask,
)
#%%
# =============================================================================
# Genuine Trimers Analysis
# =============================================================================
# Compute Functional Connectivity (FC)
fc = np.array([ts2fc(ts[animal], format_data='2D', method='pearson') 
               for animal in range(n_animals)
               ])
fc_values = fc[:, fc_idx[:, 0], fc_idx[:, 1]]

# Genuine trimers: MC_{ir,jr} > FC_{i,j}
trimers_leaves_idx = fc_reg_idx[mc_nplets_index > 0]
fc_trimers_leaves_idx = np.array([np.unique(tri_idx.flatten()) 
                                  for tri_idx in trimers_leaves_idx
                                  if len(np.unique(tri_idx.flatten())) == 3
                                  ])

#fc_trimers_leaves_idx = fc_trimers_leaves_idx[~np.isnan(fc_trimers_leaves_idx).any(axis=1)]
# Get the values of FC for the leaves of the trimers
fc_leaves_values = fc[:, fc_trimers_leaves_idx[:, 0] - 1, fc_trimers_leaves_idx[:, 1] - 1]
trimers_genuine_mc_root_fc_leaves = (mc_val[:, mc_nplets_index > 0] > fc_leaves_values)

# Genuine trimers: MC_{ir,jr} > mean(dFC_{i,j})
dfc_stream = np.array([
    ts2dfc_stream(ts[animal], window_size, lag=lag, format_data='3D', method='pearson')
    for animal in range(n_animals)
])
dfc_leaves_values = dfc_stream[:, fc_trimers_leaves_idx[:, 0] - 1, fc_trimers_leaves_idx[:, 1] - 1]
dfc_leaves_values_mean = np.mean(dfc_leaves_values, axis=-1)
trimers_genuine_mc_root_dfc_leaves = (mc_val[:, mc_nplets_index > 0] > dfc_leaves_values_mean)
#%%
# =============================================================================
# Plot Results
# =============================================================================
plt.figure(figsize=(12, 8))
plt.subplot(311)
plt.scatter(
    np.sum(trimers_genuine_mc_root_fc_leaves, axis=0) / n_animals, 
    np.sum(trimers_genuine_mc_root_dfc_leaves, axis=0) / n_animals,
    alpha=0.4, s=3
)
plt.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=1)
plt.xlabel(r'$MC_{ir,jr} > FC_{i,j}$')
plt.ylabel(r'$MC_{ir,jr} > mean(dFC_{i,j})$')
plt.tight_layout()
plt.show()
# %%

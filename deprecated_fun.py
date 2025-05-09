
#deprecated functions

def extract_hash_numbers(filenames, prefix='lot3_'):
    """Extract hash numbers from filenames based on a given prefix."""
    hash_numbers    = [int(name.split(prefix)[-1][:4]) for name in filenames if prefix in name]
    return hash_numbers

def get_mc_region_identities(fc_idx, mc_idx, sort_ref):
    aux_fc = fc_idx[sort_ref]
    fc_reg_idx = aux_fc[mc_idx]  # shape: (n_mc, 2, 2)
    mc_reg_idx = fc_reg_idx.reshape(-1, 4).T  # shape: (4, n_mc)
    return mc_reg_idx, fc_reg_idx
def fun_mc_viscocity(data):
    """
    Compute viscocity from array of trials and their MC, the first dimension must be the trials

    Parameters
    ----------
    data : N,M,M np.array 
        the trials MC.

    Returns
    -------
    mc_viscocity_val : N, MC neg values
        The first dimesion is trials, the next is variable and are list of the negative values.
    mc_viscocity_mask : M,M bool
        The mask is a boolean of the data that is true for negative value.
    """
    n_trials = data.shape[0]
    
    data = copy.deepcopy(data)
    mc_viscocity_mask = (data<0)
    mc_viscocity_val = np.array([data[i,mc_viscocity_mask[i]] for i in range(n_trials)] ,dtype='object')
    return mc_viscocity_val, mc_viscocity_mask
def contingency_matrix_fun(n_runs, mc_data, gamma_range=10, gmin= 0.8, gmax=1.3, cache_path=None, ref_name='', n_jobs=-1):
    """
    Compute or load a contingency matrix from community detection runs using joblib and vectorized agreement matrix.
    """

    n_nodes = mc_data.shape[0]
    gamma_mod = np.linspace(gmin, gmax, gamma_range)
    
    if cache_path:
        cache_dir = Path(cache_path)
        cache_dir.mkdir(parents=True, exist_ok = True)
        full_cache_path = cache_dir / f'contingency_matrix_ref={ref_name}_regions={n_nodes}_nruns={n_runs}_gamma_repetitions={gamma_range}'
        if full_cache_path.exists():
            with full_cache_path.open('rb') as f:
                print(f"[cache] Loading contingency matrix from {full_cache_path}")
                return pickle.load(f)
    else:
        full_cache_path = None

    contingency_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    gamma_qmod_val = np.zeros((gamma_range, n_runs), dtype=np.float64)
    gamma_agreement_mat = np.zeros((gamma_range, n_nodes, n_nodes), dtype=np.float64)

    for idx, gamma in enumerate(tqdm(gamma_mod, desc="Gamma values")):
        # Louvain with per-run progress bar
        results = list(tqdm(
            Parallel(n_jobs=n_jobs)(
                delayed(_run_louvain)(mc_data, gamma) 
                for _ in range(n_runs)
            ),
            total=n_runs,
            desc=f"Gamma {gamma:.2f}"
        ))

        communities, modularities = zip(*results)
        communities = np.array([np.array(c) for c in communities])
        gamma_qmod_val[idx] = modularities

        # Efficient agreement accumulation
        agreement =_build_agreement_matrix(communities)
        gamma_agreement_mat[idx] = agreement

        contingency_matrix += agreement
        

    contingency_matrix /= (n_runs * gamma_range)

    # Save to cache
    if full_cache_path is not None:
        with full_cache_path.open('wb') as f:
            pickle.dump((contingency_matrix, gamma_qmod_val, gamma_agreement_mat), f)
            print(f"[cache] Saved to {full_cache_path}")

    return contingency_matrix, gamma_qmod_val, gamma_agreement_mat


def fun_allegiance_communities_old(mc_data, n_runs=1000, gamma_pt=100, ref_name=None, save_path=None, n_jobs=-1):
    """
    Compute allegiance communities from a single or multiple mc matrices.
    Parameters:
        mc_data: 2D or 3D ndarray
            Meta-connectivity matrix or matrices.
        n_runs: int
            Number of Louvain runs per gamma value.
        gamma_pt: int
            Number of gamma values to test.
        ref_name: str
            Reference name for saving results.
        save_path: Path
            Directory to save results.
        n_jobs: int
            Number of parallel jobs.
    Returns:
        communities: ndarray
            Community labels for nodes.
        sort_idx: ndarray
            Sorting indices for nodes based on communities.
        contingency_matrix: ndarray
            Contingency matrix from Louvain runs.
    """
    # Load from the cache if the file already exists    
    def process_single(mc_matrix):#, n_runs = 10, gamma_pt = 10, ref_name='', save_path=None, n_jobs=-1): # gamma number of points in the defined range
        #allegiance index, argsort, Q value
        communities, sort_idx, _, contingency = allegiance_matrix_analysis(mc_matrix, 
                                                                           n_runs=n_runs, 
                                                                           gamma_pt=gamma_pt, 
                                                                           cache_path=save_path, 
                                                                           ref_name=ref_name, 
                                                                           n_jobs=n_jobs,
                                                                           )
        return communities, sort_idx, contingency
        

    if mc_data.ndim == 3:
        # Compute multiple allegiance communities 
        allegiances = []
        for i in range(mc_data.shape[0]):
            allegiance, _, _ = process_single(mc_data[i])
            allegiances.append(allegiance)
        mean_allegiance = np.mean(allegiances, axis=0)
        communities, sort_idx, contingency = process_single(mean_allegiance)
    elif mc_data.ndim == 2:
        communities, sort_idx, contingency = process_single(mc_data)
    else:
        raise ValueError("Input mc_data must be 2D or 3D.")
    
    communities = communities[sort_idx]

    if save_path and ref_name:
        np.savez_compressed(
            Path(save_path) / f"allegiance_{ref_name}.npz",
            communities=communities,
            sort_idx=sort_idx,
            contingency=contingency
        )

    return communities, sort_idx, contingency


# def compute_trimers_identity_old(regions):
#     """
#     Compute the indices of trimers in the meta-connectivity matrix.
#     A trimer is defined as a set of three unique nodes among the four defining a meta-connection.
#     The function returns the indices of the trimers, their region identities, and the apex node.
#     Parameters:
#     ----------
#     regions : int
#         The number of regions in the functional connectivity matrix.
#     Returns:
#     -------
#     trimer_idx : (2, M) ndarray
#         The indices of the trimers in the meta-connectivity matrix.
#     trimer_reg_id : (4, M) ndarray
#         The region identities of the trimers.
#     trimer_apex : (M,) ndarray
#         The apex node of each trimer.
#     """

#     fc_idx, mc_idx = get_fc_mc_indices(regions)

#     mc_reg_idx, _ = get_mc_region_identities(fc_idx, mc_idx)
#     mc_reg_idx = mc_reg_idx.T#, mc_ref_allegiance_sort)

#     # Find trimers: exactly 3 unique nodes among the 4 defining a meta-connection
#     unique_counts = np.array([len(np.unique(row)) for row in mc_reg_idx])
#     trimer_mask = unique_counts == 3
#     trimer_idx = mc_idx[trimer_mask].T
#     trimer_reg_id = mc_reg_idx[trimer_mask]

#     # Find apex node (node that appears twice)
#     trimer_apex = np.array([
#         np.unique(row, return_counts=True)[0][
#             np.unique(row, return_counts=True)[1] > 1
#         ][0] if len(np.unique(row, return_counts=True)[0][
#             np.unique(row, return_counts=True)[1] > 1]) > 0 else np.nan
#         for row in trimer_reg_id
#     ])

#     return trimer_idx, trimer_reg_id, trimer_apex

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
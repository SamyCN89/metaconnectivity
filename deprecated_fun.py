
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

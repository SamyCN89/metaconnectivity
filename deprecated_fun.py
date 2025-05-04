
#deprecated functions

def extract_hash_numbers(filenames, prefix='lot3_'):
    """Extract hash numbers from filenames based on a given prefix."""
    hash_numbers    = [int(name.split(prefix)[-1][:4]) for name in filenames if prefix in name]
    return hash_numbers


import numpy as np
from scipy.stats import hypergeom


def compute_page(signature, aging_up, aging_down, background_size):

    signature = set(signature)

    up_overlap = len(signature & set(aging_up))
    down_overlap = len(signature & set(aging_down))

    p_up = hypergeom.sf(
        up_overlap - 1,
        background_size,
        len(aging_up),
        len(signature)
    )

    p_down = hypergeom.sf(
        down_overlap - 1,
        background_size,
        len(aging_down),
        len(signature)
    )

    return -np.log10(p_up * p_down + 1e-300)

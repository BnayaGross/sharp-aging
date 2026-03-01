import numpy as np


def compute_page(
    drug_up,
    drug_down,
    aging_up,
    aging_down,
):
    """
    Compute raw pAGE alignment score.

    Returns a value in [-1, 1]:

        +1  -> maximal reversal of aging signature
         0  -> no relationship
        -1  -> drug mimics aging

    Parameters
    ----------
    drug_up : iterable
        Genes upregulated by drug
    drug_down : iterable
        Genes downregulated by drug
    aging_up : iterable
        Aging-up genes
    aging_down : iterable
        Aging-down genes
    """

    drug_up = set(drug_up)
    drug_down = set(drug_down)
    aging_up = set(aging_up)
    aging_down = set(aging_down)

    # --- reversal (desired effect) ---
    reversal = (
        len(drug_down & aging_up) +
        len(drug_up & aging_down)
    )

    # --- reinforcement (undesired) ---
    reinforcement = (
        len(drug_up & aging_up) +
        len(drug_down & aging_down)
    )

    total = reversal + reinforcement

    # avoid division by zero
    if total == 0:
        return 0.0

    page_score = (reversal - reinforcement) / total

    return float(page_score)

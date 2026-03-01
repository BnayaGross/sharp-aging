from sharp.core.proximity import proximity
from sharp.expression.page import compute_page


def run_sharp(
    graph,
    disease_genes,
    drug_targets,
    drug_up,
    drug_down,
    aging_up,
    aging_down,
    distance_matrix,
):

    prox = proximity(
        graph,
        drug_targets,
        disease_genes,
        distance_matrix,
        n_iter=n_random
    )

    proximity_score = prox["z_score"]

    page_score = compute_page(
        drug_up,
        drug_down,
        aging_up,
        aging_down
    )


    return {
        "network_proximity_z": proximity_score,
        "pAGE_score": page_score,
    }

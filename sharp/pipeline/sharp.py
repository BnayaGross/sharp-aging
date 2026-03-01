from sharp.core.proximity import proximity
from sharp.expression.page import compute_page


def run_sharp(
    graph,
    disease_genes,
    drug_targets,
    expression_signature,
    aging_up,
    aging_down,
    distance_matrix,
    n_random=1000
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
        expression_signature,
        aging_up,
        aging_down,
        len(graph.nodes)
    )


    return {
        "network_proximity_z": proximity_score,
        "pAGE_score": page_score,
    }

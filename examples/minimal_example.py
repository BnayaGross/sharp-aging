import sharp
import networkx as nx

# toy network
G = nx.karate_club_graph()

# disease genes
disease = {0, 1, 2}

# drug targets
targets = {10, 11}

# drug perturbation signature
drug_up = {3, 4}
drug_down = {5}

# aging signature
aging_up = {3, 6}
aging_down = {4, 7}

# distance matrix
D = dict(nx.all_pairs_shortest_path_length(G))

# run SHARP
res = sharp.run_sharp(
    graph=G,
    disease_genes=disease,
    drug_targets=targets,
    drug_up=drug_up,
    drug_down=drug_down,
    aging_up=aging_up,
    aging_down=aging_down,
    distance_matrix=D,
)

print(res)

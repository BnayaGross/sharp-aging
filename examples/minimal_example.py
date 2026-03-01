import sharp
import networkx as nx

G = nx.karate_club_graph()

disease = {0,1,2}
targets = {10,11}

drug_up = {3,4}
drug_down = {5}
aging_up = {3,4}
aging_down = {6,7}

D = dict(nx.all_pairs_shortest_path_length(G))

res = sharp.run_sharp(
    graph=G,
    disease_genes=disease,
    drug_targets=targets,
    drug_up=drug_up,
    drug_down=drug_down,
    aging_up=aging_up,
    aging_down=aging_down,
    distance_matrix=D
)

print(res)

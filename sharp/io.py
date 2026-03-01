import pandas as pd
import networkx as nx

def load_interactome(path, source="source", target="target"):
    df = pd.read_csv(path)
    G = nx.from_pandas_edgelist(df, source, target)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G

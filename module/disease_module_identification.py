from multiprocessing import Process, Value, Array, Manager
import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import spdiags
from scipy.stats import norm, hypergeom
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import KFold
from statsmodels.stats.multitest import multipletests
from statistics import mean, stdev
from Diamond import DIAMOnD
import random
import xml.etree.ElementTree as ET
import requests
import json
from Bio import Entrez
Entrez.email = 'farzaneh.nasirian@gmail.com'
from IPython.display import Markdown, display
from tqdm import tqdm
import matplotlib.pyplot as plt



def lcc_func(G, nodes):
    nodes = G.nodes() & nodes
    return( max(nx.connected_components(G.subgraph(nodes)), key=len) if nodes else set() )


def LCC_zscore(G, nodes, degree_preserving=True):
    nodes = G.nodes() & nodes
    x = []
    LCC_size = len(lcc_func(G, nodes))
    if degree_preserving == False:
        random_nodes = [random.sample(G.nodes(), len(nodes)) for i in range(1000)]
    else:
        random_nodes = get_random_nodes(nodes, set(), G, n_random=100, min_bin_size=100)   
    for i in range(1000): x.append(float(len(lcc_func(G, random_nodes[i]))))

    Ave = float(mean(x))
    Std = stdev(x)
    if Std == 0:
        Std = 0.001
    z_score = (LCC_size-Ave)/Std
    #p_value = norm.sf(z_score) 
    p_value = float(len([d for d in x if d >= LCC_size])/1000)

    return((p_value, z_score, LCC_size, Ave))

def LCC_zscore_new(G, nodes, degree_preserving=True):
    nodes = G.nodes() & nodes
    x = []
    LCC_size = len(lcc_func(G, nodes))
    if degree_preserving == False:
        random_nodes = [random.sample(G.nodes(), len(nodes)) for i in range(1000)]
    else:
        random_nodes = get_random_nodes(nodes, set(), G, n_random=1000, min_bin_size=100)   
    for i in tqdm(range(1000)): x.append(float(len(lcc_func(G, random_nodes[i]))))

    Ave = float(mean(x))
    Std = stdev(x)
    if Std == 0:
        Std = 0.001
    z_score = (LCC_size-Ave)/Std
    #p_value = norm.sf(z_score) 
    p_value = float(len([d for d in x if d >= LCC_size])/1000)

    return((p_value, z_score, LCC_size, Ave, x))

def LCC_zscore_par(G, nodes, par, degree_preserving=True):
    nodes = G.nodes() & nodes
    x = []
    LCC_size = len(lcc_func(G, nodes))
    if degree_preserving == False:
        random_nodes = [random.sample(G.nodes(), len(nodes)) for i in range(10)]
    else:
        random_nodes = get_random_nodes_par(nodes, set(), G, par, n_random=10, min_bin_size=100)  
        print(len(random_nodes))
    for i in tqdm(range(par*10)): x.append(float(len(lcc_func(G, random_nodes[i]))))

    Ave = float(mean(x))
    Std = stdev(x)
    if Std == 0:
        Std = 0.001
    z_score = (LCC_size-Ave)/Std
    #p_value = norm.sf(z_score) 
    p_value = float(len([d for d in x if d >= LCC_size])/(10*par))

    return((p_value, z_score, LCC_size, Ave, x))

def LCC_zscore_high(G, nodes, degree_preserving=True):
    nodes = G.nodes() & nodes
    x = []
    LCC_size = len(lcc_func(G, nodes))
    if degree_preserving == False:
        random_nodes = [random.sample(G.nodes(), len(nodes)) for i in range(10000)]
    else:
        random_nodes = get_random_nodes(nodes, set(), G, n_random=10000, min_bin_size=100)   
    for i in tqdm(range(10000)): x.append(float(len(lcc_func(G, random_nodes[i]))))

    Ave = float(mean(x))
    Std = stdev(x)
    if Std == 0:
        Std = 0.001
    z_score = (LCC_size-Ave)/Std
    #p_value = norm.sf(z_score) 
    p_value = float(len([d for d in x if d >= LCC_size])/10000)

    return((p_value, z_score, LCC_size, Ave, x))


def normalize_matrix(G, Type):
    
    A = nx.adjacency_matrix(G)
    n = A.shape[0]
    d = A.sum(axis = 1)
    d = np.asarray(d).flatten()
    d = np.maximum(d, np.ones(n)) #to handle zeros (isolated nodes) 
    
    if Type == 'col': 
        invd = spdiags(1.0 / d, 0, n, n)  #inverse degree diagonal mtx
        nA = A.dot(invd) 
        return(nA)
    
    elif Type == 'col_row': 
        fracD = spdiags(1/np.sqrt(d), 0, n, n)
        nA = fracD.dot(A.dot(fracD))
        return(nA)

    
def run_walk(G, int_p, A, alpha):  
    """
    int_p: transition probabilities at time zero
    A: normalized adj matrix
    alpha: restart probability
    return p (stationary transition probability)
    """
    
    p = np.copy(int_p)
    res = 1
    while res > 0.000001:
        p_new = alpha * int_p + (1-alpha) * A.dot(p)
        res = np.linalg.norm(p_new - p, 1)
        p = p_new
        
    return(np.squeeze(np.asarray(p)))


def get_random_nodes(nodes, avoid_nodes, network, n_random=1000, min_bin_size=100):
    bins = get_degree_binning(network, min_bin_size) 
    nodes_random = pick_random_nodes_matching_selected(network, bins, nodes, avoid_nodes, n_random)
    
    return(nodes_random)

def get_random_nodes_par(nodes, avoid_nodes, network, par, n_random=1000, min_bin_size=100):
    tt = []
    with Manager() as manager:
        values = manager.list()
        bins = get_degree_binning(network, min_bin_size) 
        #nodes_random = pick_random_nodes_matching_selected_test(network, bins, nodes, avoid_nodes, n_random,values)
        processes = []
        for i in range(par):
            p = Process(target=pick_random_nodes_matching_selected_test, args=(network, bins, nodes, avoid_nodes, n_random,values))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        print(len(values))
        tt = list(values)
    return(tt)


def get_degree_binning(g, bin_size):
    degree_to_nodes = {}
    for node, degree in g.degree(): 
        degree_to_nodes.setdefault(degree, []).append(node)
    values = sorted(degree_to_nodes)
    bins = []
    i = 0
    while i < len(values):
        low = values[i]
        val = degree_to_nodes[values[i]]
        while len(val) < bin_size:
            i += 1
            if i == len(values):
                break
            val.extend(degree_to_nodes[values[i]])
        if i == len(values):
            i -= 1
        high = values[i]
        i += 1 
        #print i, low, high, len(val) 
        if len(val) < bin_size:
            low_, high_, val_ = bins[-1]
            bins[-1] = (low_, high, val_ + val)
        else:
            bins.append((low, high, val))
            
    return(bins)


def get_degree_equivalents(seeds, bins, g):
    seed_to_nodes = {}
    for seed in seeds:
        d = g.degree(seed)
        for l, h, nodes in bins:
            if l <= d and h >= d:
                mod_nodes = list(nodes)
                mod_nodes.remove(seed)
                seed_to_nodes[seed] = mod_nodes
                break
    
    return(seed_to_nodes)


def pick_random_nodes_matching_selected(network, bins, nodes_selected, avoid_nodes, n_random):
    values = []
    nodes = network.nodes()
    for i in tqdm(range(n_random)):
        nodes_random = set()
        node_to_equivalent_nodes = get_degree_equivalents(nodes_selected, bins, network)
        for node, equivalent_nodes in node_to_equivalent_nodes.items():
            chosen = random.choice(equivalent_nodes)
            for k in range(20): # Try to find a distinct node (at most 20 times)
                if chosen in (nodes_random|avoid_nodes):
                    chosen = random.choice(equivalent_nodes)
                else:
                    break
            nodes_random.add(chosen)
        values.append(list(nodes_random))
        
    return(values)


def pick_random_nodes_matching_selected_test(network, bins, nodes_selected, avoid_nodes, n_random, values):
    nodes = network.nodes()
    for i in tqdm(range(n_random)):
        nodes_random = set()
        node_to_equivalent_nodes = get_degree_equivalents(nodes_selected, bins, network)
        for node, equivalent_nodes in node_to_equivalent_nodes.items():
            chosen = random.choice(equivalent_nodes)
            for k in range(20): # Try to find a distinct node (at most 20 times)
                if chosen in (nodes_random|avoid_nodes):
                    chosen = random.choice(equivalent_nodes)
                else:
                    break
            nodes_random.add(chosen)
        values.append(list(nodes_random))
        
    


def rwr_ranks(G, seeds, Type, r, normalize, method):
        
    A = normalize_matrix(G, Type)
    n = G.order()
    nodes = pd.DataFrame(np.zeros((1,n)), columns=G.nodes())
    
    def transition_prob(G, seed_set, r):
        nodes.loc[0] = 0
        nodes.loc[0, list(seed_set)] = 1
        int_p = nodes.loc[0].values
        return(run_walk(G, int_p, A, r))
    
    def stand(x): return( (x[2] - mean(x[3:]))/stdev(x[3:]) )   #used sample standard deviation

    df = pd.DataFrame()
    df['GeneId'] = G.nodes()
    df['seed'] = df.GeneId.apply(lambda x: True if x in seeds else False)
    df['tprob'] = transition_prob(G, seeds, r)

    if normalize == True: 
        #reference model: adjust probabilities by seed degrees
        ref_mdl = dict()
        rand_nodes = get_random_nodes(seeds, seeds, G, n_random=100)
        i = 0
        for node_set in rand_nodes:
            ref_mdl['round_'+str(i)] = transition_prob(G, node_set, r)
            i += 1 
     
        df = pd.concat([df, pd.DataFrame(ref_mdl)], axis=1)
        df['z_value'] = df.apply(stand, axis=1)  
        df['p_value'] = df.z_value.apply(lambda x: norm.sf(x)) #empirical p-value
        df = df[df.seed == False]
        df['adj_p_value'] = multipletests(pvals=df.p_value, method=method)[1]
        df = df[['GeneId', 'tprob', 'z_value', 'p_value', 'adj_p_value']]
 
        return(df)
        
    else:
        df = df[df.seed == False]
        return(df.drop('seed', axis=1))


def gene_ranks(G, seeds, rank_mdl, Type='col_row', r=0.2, normalize=False, method='bonferroni', max_iter=1000):

    """
    This version supports three local-connectivity-based ranking approaches; RWR, DAIMAOnD, Guilt-by-Association (GA)
    inputs:
        G: graph-like object
        seeds: list of gene sets used for initiating rwr or diamond algorithms. It contains two columns "Set" and "GeneId"
        rank_mdl: gets two values; rwr and diamond
        Type: gets two values; "col" and "col_row"
            - 'col': adj matrix will be normalized by degree of source node. 
            - 'col_row': adj matrix will be normalized by degree of both source and target.
        r: restart probability. Likelihood of re-visiting a seed node
            normalize: set to "True" to adjust transition probabilities by 
            a reference set of randomly selected nodes with degree distribution similar to seed genes. 
            Otherwise, genes will be ranked by transition probabilities in steady state. 
        method: method for p-value correction
        max_iter: number of ranked candidates. At each iteration, the top ranked gene is added to the initial seed set. 
            The new seed set is used for ranking in the next iteration.


    outputs:
        Set: name of seed set
        GeneId: gene symbol
        rank_val: transition probability/connectivity significance (-log10) to the ranked gene 
        itr: iteration index that varies between zero and max_iter. At iteration zero, GeneIds are initial seeds  
    """

    if rank_mdl == 'rwr': 
        seeds['itr'] = 0
        criterion = 'adj_p_value' if normalize == True else 'tprob'    
        ascending = True if normalize == True else False
        for i in tqdm(range(1, max_iter+1)):
            temp = []
            for s in seeds.Set.unique():
                nodes = set(seeds[seeds.Set == s]['GeneId'])
                res = rwr_ranks(G, nodes, Type=Type, r=r, normalize=normalize, method=method)
                g, p = res[~res.GeneId.isin(nodes)].sort_values(criterion, ascending=ascending).iloc[0][:2].tolist()
                temp.append([s, g, p, i])
            seeds = pd.concat([seeds, pd.DataFrame(temp, columns=['Set', 'GeneId', 'rank_val', 'itr'])])
        seeds = seeds.reset_index(drop=True)
        return(seeds)

    elif rank_mdl == 'diamond':
        seeds['itr'] = 0
        for s in tqdm(seeds.Set.unique()):
            temp = []
            nodes = seeds[seeds.Set == s].GeneId.unique()
            res = DIAMOnD(G, nodes, max_iter, alpha=1)
            for i in range(max_iter): temp.append([s, res[i][0], -np.log10(res[i][-1][0]), i+1])
            seeds = pd.concat([seeds, pd.DataFrame(temp, columns=['Set', 'GeneId', 'rank_val', 'itr'])])
        seeds = seeds.reset_index(drop=True)
        return(seeds)

    elif rank_mdl == 'GA':
        seeds['itr'] = 0
        for i in tqdm(range(1, max_iter)):
            temp = []
            for s in seeds.Set.unique():
                nodes = seeds[seeds.Set == s].GeneId.unique()
                genes = set(random.sample(set.union(*[set(G.neighbors(g)) for g in nodes]), 1))
                for g in genes: temp.append([s, g, 1, i])
            seeds = pd.concat([seeds, pd.DataFrame(temp, columns=['Set', 'GeneId', 'rank_val', 'itr'])])
        seeds = seeds.reset_index(drop=True)
        return(seeds)



def cv_eval(G, seeds, rank_mdl, genomic_distance, n_split=5):

    """
    outputs:
        Set: name of seed set
        fold: fold index
        auc: area under the ROC curve corresponding to each fold
        aup: area under the percision-recall curve corresponding to each fold
        median: median rank of holdout genes
        ranks: rank of holdout genes
    """
    
    CV = []
    for s in seeds.Set.unique():
        X = seeds[(seeds.Set == s)].GeneId.values
        f_iter = 0
        for train_idx, test_idx in KFold(n_splits=n_split, shuffle=True).split(X):
            #split the data
            holdout = X[test_idx]
            train = X[train_idx]
            max_iter = G.order() - len(train)
            cv_res = gene_ranks(G, seeds, rank_mdl, method='bonferroni', max_iter=max_iter)

            #undersampling negative class 
            subsamp = set().union(holdout.tolist(), genomic_distance[genomic_distance.source.isin(holdout)].target.tolist())
            subsamp = subsamp - set(train)
            subsamp = cv_res[cv_res.GeneId.isin(subsamp)].copy()

            #assign class labels
            subsamp['holdout'] = subsamp.GeneId.apply(lambda x: 1 if x in holdout else 0)
            subsamp['fold'] = f_iter
            subsamp['model'] = s
            f_iter += 1
            CV.extend(subsamp.values)
    CV = pd.DataFrame(CV, columns=subsamp.columns)
    
    cperf = []
    for m in CV.model.unique():
        for f in CV.fold.unique():
            subsamp = CV[(CV.model == m) & (CV.fold == f)].sort_values('rank_val', ascending=False).reset_index(drop=True)
            auc_score = roc_auc_score(subsamp.holdout, subsamp.rank_val)   #auc_roc
            precision, recall, thr = precision_recall_curve(subsamp.holdout, subsamp.rank_val)
            aup_score = auc(recall, precision)  #auc_pr
            rs = subsamp[subsamp.holdout == 1].index.tolist() #rank list
            med = np.median(rs)
            cperf.append([m, f, auc_score, aup_score, med, rs])
    cperf = pd.DataFrame(cperf, columns=['Set', 'fold', 'auc', 'aup', 'median', 'ranks'])
    
    return(cperf)



def lcc_eval(G, seeds, step, max_iter=1000, method='bonferroni'):

    """
    step: fixed-size gap between two consecutive iteration
    seeds: output of gene_ranks function or any other ranking approaches. 
        It is composed of four columns; Set, GeneId, rank_val, and itr.

    outputs:
        g_order: number of selected seeds at iteration i
        lcc_order: order of lcc induced by seeds at iteration i
        rlcc: relative order of lcc at iteration i
        recall_lcc: fraction of initial seeds (e.g., causal genes) present in the ranked lcc at iteration i
        recall_lcc_random: fraction of initial seeds (e.g., causal genes) present in randomly constructed lcc at iteration i
        p_value: significance of the ranked lcc at iteration i
    """

    df = seeds[seeds.itr == 0].groupby('Set')['GeneId'].apply(set).reset_index(name='inti_seeds')
    def lcc_recall(x):
        return( len(x[4] & x[1])/len(x[1]) )

    lcc_data = []
    for i in tqdm(range(0, max_iter, step)):
        df['GeneId'] = seeds[seeds.itr <= i].groupby('Set')['GeneId'].apply(set)
        df['g_order'] = df.GeneId.apply(lambda x: len(x))
        df['lcc'] = df.GeneId.apply(lambda x: lcc_func(G, x))
        df['lcc_order'] = df.lcc.apply(lambda x: len(x))
        df['rlcc'] = df.lcc_order/df.g_order
        df['recall_lcc'] = df.apply(lcc_recall, axis=1)
        df['lcc'] = df.inti_seeds.apply(lambda x: lcc_func(G, x | set(random.sample(set.union(*[set(G.neighbors(g)) for g in x]), i)) ))
        df['recall_lcc_random'] = df.apply(lcc_recall, axis=1)     
        df['p_value'] = df.GeneId.apply(lambda x: LCC_zscore(G, x)[0])
        df['itr'] = i
        lcc_data.extend(df.drop(['GeneId', 'lcc'], axis=1).values)
    lcc_data = pd.DataFrame(lcc_data, columns=df.drop(['GeneId', 'lcc'], axis=1).columns)
    adjp = []
    lcc_data = lcc_data.sort_values('Set')
    for s in lcc_data.Set.unique(): adjp.extend(multipletests(lcc_data[lcc_data.Set == s].p_value, method=method)[1])
    lcc_data['adj_pvalue'] = adjp    

    return(lcc_data)



def enrich_eval(G, seeds, ref_sets, max_iter=1000, method='fdr_bh'):

    """
    seeds: output of gene_ranks function or any other ranking approaches. 
        It is composed of four columns; Set, GeneId, rank_val, and model.
    ref_sets: set of gene sets to compute enrichment of seed genes against. 
        It is composed of two columns; Set and GeneId

    outputs:
        Set: name of seed set
        ref_set: name of reference set used for enrichment 
        method: methodology used for gene ranking
        p: enrichment p-value at iteration i
        adj_p: corrected p-value
        itr: iteration index that varies between zero and max_iter. At iteration zero, GeneIds are initial seeds 
    """

    ref_dict = ref_sets.groupby('Set')['GeneId'].to_dict()

    M = G.order()

    enrich = []
    for m in tqdm(seeds.model.unique()):
        for s in seeds.Set.unique():
            df = seeds[(seeds.model == m) & (seeds.Set == s)].reset_index(drop=True).sort_values('itr')
            n = df[df.itr == 0].shape[0]
            for j in range(0, max_iter+1):
                x = set(df[j:(n+j)]['GeneId'])
                for k,v in ref_dict.items(): enrich.append([s, j, k, hypergeom.sf(len(x & v)-1, M, len(v), len(x)), m])
    enrich = pd.DataFrame(enrich, columns=['Set', 'itr', 'ref_set', 'p', 'method'])
    enrich = enrich.groupby(['Set', 'ref_set', 'method'])['p'].apply(list).reset_index()
    enrich['adj_p'] = enrich.p.apply(lambda x: multipletests(x, method=method)[1])
    enrich['itr'] = [np.arange(0,max_iter+1) for i in enrich.index]
    enrich = enrich.explode(['p', 'adj_p', 'itr']).reset_index(drop=True)

    return(enrich)



def common_rank_eval(G, seeds, interval, max_iter, method='fdr_bh'):
    
    """
    application: compare ranked genes by two methodologies
    interval: an interval for comparison of ranked genes
    seeds: output of gene_ranks function or any other ranking approaches. 
        It is composed of four columns; Set, GeneId, itr, and method. At iteration zero, GeneIds are initial seeds which are
        identical in all methods
    """

    m_i, m_j = seeds.method.unique()
    M = G.order()

    enrich = []
    for s in tqdm(seeds.Set.unique()):
        for i in range(1, max_iter-interval+2):
            A = set(seeds[(seeds.Set == s) & (seeds.method == m_i) & (seeds.itr >= i) & (seeds.itr < i+interval)]['GeneId'])
            B = set(seeds[(seeds.Set == s) & (seeds.method == m_j) & (seeds.itr >= i) & (seeds.itr < i+interval)]['GeneId'])
            p = hypergeom.sf(len(A&B)-1, M, len(A), len(B))
            enrich.append([s, i+interval-1, len(A&B), len(A&B)/len(A|B), p])
        enrich = pd.DataFrame(enrich, columns=['Set', 'itr', 'cg_count', 'JS', 'p']) 
        enrich['adj_p'] = enrich.groupby(['Set'])['p'].apply(list).apply(
            lambda x: multipletests(x, method=method)[1]).explode().tolist()

    return(enrich)


def pdis(A, B, prop, dist, identical=True):

    """
    application: compute pairwise distance between nodes of sets A and B
    prop: nodes' properties table
    dist: pairwise distance matrix
    identical: it is True if A==B
    """

    y = prop[prop.GeneId.isin(B)].id.tolist()
    d_s = []
    if identical:
        for g in A:
            a = np.take(dist[g], y)
            d_s.append(a[a != 0].min())    
        return(d_s)
    else:
        for g in A:
            a = np.take(dist[g], y)
            d_s.append(a.min())    
        return(d_s)



def local_modularity (G, cluster): 
    
    cluster = set(cluster)

    numerator = 0
    denominator = 0
    for n in cluster:
        temp = set(G.neighbors(n)) - cluster
        if bool(temp):
            numerator += len(temp)
            denominator += G.degree(n)

    return (1-numerator/denominator) 

    

def Infomap(G):

    """
    Partition network with the Infomap algorithm.
    Annotates nodes with 'community' id.
    """
    import infomap

    im = infomap.Infomap("--two-level")
    im.add_networkx_graph(G)
    im.run()

    print(f"Found {im.num_top_modules} modules with codelength: {im.codelength}")

    communities = im.get_modules()
    return(communities)


def enrichr(Input, lib):
    #Input: a dataframe of sets that are of interst for enrichement analysis. columns are "set" and "genes" 
    #lib: gene set lib for enrichement analysis
    #df: dataframe of enrichement results 

    df = [] 
    for i in Input.index:
        lisT = Input.loc[i].tolist()
        ENRICHR_URL = 'http://maayanlab.cloud/Enrichr/addList'
        genes_str = '\n'.join(lisT[1])  
        description = 'None'
        payload = {
            'list': (None, genes_str),
            'description': (None, description)
        }

        response = requests.post(ENRICHR_URL, files=payload)
        if not response.ok:
            raise Exception('Error analyzing gene list')

        ENRICHR_URL = 'http://maayanlab.cloud/Enrichr/enrich'
        query_string = '?userListId=%s&backgroundType=%s'
        user_list_id = json.loads(response.text)['userListId']

        response = requests.get(
            ENRICHR_URL + query_string % (user_list_id, lib)
        )
        if not response.ok:
            raise Exception('Error fetching enrichment results')
            
        data = json.loads(response.text)  
        for value in list(data.values())[0]:
            df.append([lisT[0], value[1], value[6]])

    df = pd.DataFrame(df, columns=['set_A', 'set_B', 'adj_p_value'])
    return(df)


def drugtarget(xml_file):

    tree = ET.parse(xml_file)
    root = tree.getroot()
        
    drug_vec = []
    for drug in root:

        temp_vec = [np.nan]*5
        
        ####drug type
        temp_vec[0] = drug.attrib['type']
        prime = 0
        for child in drug:
            
            ####DrugBank ID
            if (child.tag[24:] == 'drugbank-id' and prime == 0):
                temp_vec[1] = child.text
                prime = 1
                
            ####ChEMBL ID
            if child.tag[24:] == 'external-identifiers':
                temp_vec[2] = set()
                for prop in child:
                    for item in prop:
                        code, source = (prop.find('{http://www.drugbank.ca}resource'),
                                        prop.find('{http://www.drugbank.ca}identifier'))
                        if code.text == 'ChEMBL':
                            temp_vec[2].add(source.text)
                
            ####Drug name    
            if child.tag[24:] == 'name':
                temp_vec[3] = child.text.lower()
            
            ####Drug affected-organism
            if child.tag[24:] == 'affected-organisms': 
                temp = []
                for brand in child:
                    temp.append(brand.text)
                temp_vec[4] = temp
            
            ####Drug targets
            if child.tag[24:] in ['targets', 'enzymes', 'carriers', 'transporters']:
                for target in child:
                    char = [float('nan')]*4
                    char[0] = child.tag[24:] 
                    for profile in target: 
                        if profile.tag[24:] == 'known-action':
                            char[1] = profile.text        
                        if profile.tag[24:] == 'polypeptide':   #to get the gene encoding the target
                            for info in profile:
                                if info.tag[24:] == 'gene-name':
                                    char[2] = info.text  
                        if profile.get('id'):                   #uniprot id for the target
                            char[3] = profile.get('id')           
                    drug_vec.append(temp_vec + char)

    columns = ['type', 'dbId', 'chemblId', 'name', 'organism', 'target_type', 'action', 'GeneId', 'UniProt']
    DrugBank = pd.DataFrame(drug_vec, columns=columns) 
    DrugBank = DrugBank.mask(DrugBank.applymap(str).eq('[]'))

    return(DrugBank) 


def display_color(colors):
    """
    colors: list of hex color codes
    """

    display(Markdown('<br>'.join(
        f'<span style="font-family: monospace">{color} <span style="color: {color}">████████</span></span>'
        for color in colors.values()
    )))


def autolabel(rects):    #Function: Add a text label above each bar in *rects*, displaying its height.
    for rect in rects:
        height = rect.get_height()
        plt.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 1),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


#variant-gene physical mapping with Entrez API
#Over one billion Variant-Gene associations are collected by NCBI SNP library. 
# [Here](https://ftp.ncbi.nih.gov/snp/latest_release/JSON/) is the link to the FTP gateway to download 500G data.  
def snp_to_gene(snp_ids):
    
    df = []

    for snp_id in tqdm(snp_ids):
        record = Entrez.read(Entrez.elink(dbfrom="snp", id=snp_id.replace('rs',''), db="gene"))
        try:
            results = record[0]['LinkSetDb'][0]['Link']
            for result in results:
                gene_id = result['Id']
                handle = Entrez.esummary(db="gene", id=gene_id)
                gene_symbol = Entrez.read(handle)["DocumentSummarySet"]['DocumentSummary'][0]['Name']
                handle.close()
                df.append([snp_id, gene_id, gene_symbol])
        except:
            pass

    df = pd.DataFrame(df, columns=['snpId', 'entrezId', 'GeneId'])
    return(df)
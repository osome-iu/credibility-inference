import pandas as pd 
import numpy as np
import networkx as nx
from os.path import join 

RELATIVE_PATH = '/nfs/nfs9/home/nobackup/baotruon/cocred/cocred_jan2/exp'
# Make sure GRAPHML exists!vb
GRAPHML = 'graphml'

bipartite = 'bipartite/user_domain'
user_coshare = 'bipartite/user_user'
rt = 'user_rt/rt_edgelist'
ulabel_path = 'user_labels.csv'
dlabel_path = 'domain_labels.csv'

ulabels = pd.read_csv(join(RELATIVE_PATH,ulabel_path)) #cols: uid 	mean_score_times 	true_label
ulabels = ulabels.astype({'uid':str})

dlabels = pd.read_csv(join(RELATIVE_PATH,dlabel_path)) #cols: Domain 	Rating 	Score 	label

cols = {
    'bipartite':{'node_col':'domain', 'label_col':'label', 'ncol1':'domain', 'ncol2':'uid'},
    'rt':{ 'node_col':'uid', 'label_col':'true_label', 'ncol1':'uid1', 'ncol2':'uid2'},
    'user_coshare':{ 'node_col':'uid', 'label_col':'true_label', 'ncol1':'uid1', 'ncol2':'uid2'}
}

labeled_df_name_map={
    'bipartite': {'label':'domain_label', 'tweet_counts':'weight'},
    'rt': {'true_label_x':'uid1_label','true_label_y':'uid2_label','tweet_counts':'weight'},
    'user_coshare': {'true_label_x':'uid1_label','true_label_y':'uid2_label','cosine_sim':'weight'}
}



def label_parquet_edgelist(edgelist, labels, node_col='uid', label_col='label', ncol1='uid1', ncol2='uid2', network_type='rt'):
    #(edgelist, labels, node_col='uid', label_col='label', w_col='weight', network_type='')
    print('Making graph for node %s and node %s ..' %(ncol1, ncol2))
    w_col='weight'
    edgelist = edgelist.astype({ncol1:str, ncol2:str})
    labels = labels.astype({node_col:str})
    
    if network_type=='rt' or network_type=='user_coshare':
        network_ = pd.merge(edgelist, labels.loc[:,[node_col, label_col]], left_on= ncol1, right_on='uid', how='left')
        network_labeled = pd.merge(network_, labels.loc[:,[node_col, label_col]], left_on= ncol2, right_on='uid', how='left')
        network_labeled = network_labeled.rename(columns = labeled_df_name_map[network_type])
        network_labeled['%s_label' %ncol1] = network_labeled['%s_label' %ncol1].apply(lambda x: -1 if np.isnan(x) else x)
        network_labeled['%s_label' %ncol2] = network_labeled['%s_label' %ncol2].apply(lambda x: -1 if np.isnan(x) else x)
    else:
        #bipartite network 
        network_labeled = pd.merge(edgelist, labels.loc[:,[node_col, label_col]], on=node_col, how='left')
        network_labeled = network_labeled.rename(columns = labeled_df_name_map[network_type])
        network_labeled['%s_label' %ncol1] = network_labeled['%s_label' %ncol1].apply(lambda x: -1 if np.isnan(x) else x)
    
    G = nx.Graph()
    for idx, row in network_labeled.iterrows():
        if row[ncol1] not in set(G.nodes()):
            if network_type=='bipartite':
                G.add_node(row[ncol1], label=row['%s_label' %ncol1], partite=0)
            else:
                G.add_node(row[ncol1], label=row['%s_label' %ncol1])
        if row[ncol2] not in set(G.nodes()):
            if network_type=='bipartite':
                #uid doesn't have label
                G.add_node(row[ncol2], partite=1)
            else:
                G.add_node(row[ncol2], label=row['%s_label' %ncol2])
                
        G.add_edge(row[ncol1], row[ncol2], weight=row[w_col])
        
    return G

#TODO: Clean this
""" Filter and retain only domains that are shared by at least k users. Not recursive. 
    Can consider a recursive approach for bipartite network later"""
def prune_domainuser_graph(df,k=3):
    domain_grouped = df.groupby([domain_col])[user_col].nunique().reset_index()
    kcore = domain_grouped[domain_grouped[user_col]>3]
    kcore_domains = kcore[domain_col].values
    print('Before pruning: #users: %s, #domains: %s, #edges: %s' %(len(df[user_col].unique()), len(df[domain_col].unique()), len(df)))
    df_kcore= df[df[domain_col].isin(kcore_domains)]
    print('After pruning: #users: %s, #domains: %s,#edges: %s' %(len(df_kcore[user_col].unique()), len(df_kcore[domain_col].unique()), len(df_kcore)))
    return df_kcore


rt_df = pd.read_parquet(join(RELATIVE_PATH, rt+'.parquet'))
coshare = pd.read_parquet(join(RELATIVE_PATH, user_coshare+'.parquet'))
bi_df = pd.read_parquet(join(RELATIVE_PATH, bipartite+'.parquet'))

nets = [rt_df, coshare, bi_df]
label_dfs = [ulabels, ulabels, dlabels]
netnames = ['rt','user_coshare', 'bipartite']
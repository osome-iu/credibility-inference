""" Provides wrapper to use the Pecanpy graph embedding generation library"""

# from pecanpy import node2vec
from gensim.models import word2vec, keyedvectors
import igraph
import os
import pandas as pd


def graph_from_file(edgelist):
    """
    Make igraph instance of network
    type = ['graphml', 'graphtxt']
    """

    _, extension = os.path.splitext(edgelist)

    if extension == ".graphml":
        graph = igraph.Graph.Read_GraphML(edgelist)
    elif extension == ".gml":
        graph = igraph.Graph.Graph.Read_GML(edgelist)
    elif extension == ".txt":
        graph = igraph.Graph.Read_Ncol(edgelist, weights=True, directed=True)

    elif extension == ".parquet":
        df = pd.read_parquet(edgelist, engine="pyarrow")
        graph = igraph.Graph.TupleList(
            df.itertuples(index=False),
            directed=True,
            weights=True,  # only True if column "weight" exists
            # edge_attrs="weight", #alternative way to specify weight col
        )

    # return df with 2 cols: [vertex ID, name]
    vertex_df = graph.get_vertex_dataframe().reset_index()

    return graph, vertex_df


def format_edgelist(inf, outf=None, insep=" ", outsep="\t", reverse=False):
    """
    Convert between tab-separated and space-separated edgelist
    Reverse direction of edge if reverse=True
    """
    net = []
    with open(inf, "r") as f:
        for line in f:
            data = line.strip("\n")
            edge = data.split(insep)
            if reverse == True:
                net += [(edge[1], edge[0], float(edge[2]))]
            else:
                net += [(edge[0], edge[1], float(edge[2]))]
    if outf is not None:
        save_edgelist(net, outf, sep=outsep)
        return net
    else:
        return net


def save_edgelist(edgelist, fpath, sep="\t"):
    # edgelist: iterable of a tuple representing an edge (v1, v2, weight)
    with open(fpath, "w") as f:
        for edgetuple in edgelist:
            newline = sep.join([str(item) for item in edgetuple])
            f.write(f"{newline}\n")
    print("Finished saving edgelist!")
    return fpath


def word2vec_object_to_dict(wv_model):
    """
    Create dict from wv model
    """
    if isinstance(wv_model, word2vec.Word2Vec):
        wv_model = wv_model.wv
    elif not isinstance(wv_model, keyedvectors.KeyedVectors):
        raise ValueError(
            "embedding should be an instance of gensim.models Word2vec or Keyedvectors"
        )
    vectors = wv_model.vectors
    embeddings = {}
    for i, word in enumerate(wv_model.index_to_key):
        embeddings[word] = vectors[i]
    return embeddings


# def get_node2vec_emb(edge_fpath, graph_params=None, walk_params=None, w2v_params=None):
#     """
#     Wrapper for creating graph embedding using pecanpy.
#     For our purpose, SparseOTF mode is used (the graph is sparse but not necessarily small)
#     For more information, see: https://github.com/krishnanlab/PecanPy
#     Parameters:
#     -----------
#     - graph_params (dict): specify p, q for the biased random walk. Default if None.
#     - walk_params (dict): specify other params. Default if None.
#     - w2v_params (dict): specify params for Word2vec model. Default if None.
#     - edge_fpath: weighted edgelist, tab-separated

#     Returns:
#     -----------
#     - w2v_model (dict): the graph embedding {node_name: embedding_vector}
#     """
#     # load graph object using SparseOTF mode
#     if graph_params is None:
#         graph_params = {"p": 0.25, "q": 0.1}
#     if walk_params is None:
#         walk_params = {"num_walks": 10, "walk_length": 80, "n_ckpts": 10, "pb_len": 100}
#     if w2v_params is None:
#         w2v_params = {
#             "vector_size": 128,
#             "window": 5,
#             "min_count": 0,
#             "sg": 1,
#             "workers": 3,
#             "epochs": 5,
#         }
#     # generate random walks
#     pecan_obj = node2vec.SparseOTF(**graph_params, workers=1, verbose=True)
#     pecan_obj.read_edge(edge_fpath, weighted=True, directed=False)

#     # Note: simulate walks parameters differ for different versions of peanpy
#     # only for new ver of pecan: n_ckpts: number of checkpoints; pb_len: progress bar length for monitoring

#     # use random walks to train embeddings
#     walks = pecan_obj.simulate_walks(**walk_params)
#     # previous pecanpy version:
#     # walks = g.simulate_walks(num_walks=10, walk_length=80)

#     w2v_model = Word2Vec(walks, **w2v_params)
#     return w2v_model

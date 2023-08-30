"""
Get the Giant CC of the network. 
Option to sample edges 
NOTE: this is for undirected graph
Usage: `python3 workflow/preprocess/filter_network.py undirected cosharing_edgelist.txt derived_network`
"""
import igraph as ig
import networkx as nx
import sys
import pandas as pd
import os
from copy import deepcopy
import numpy as np
from tqdm import tqdm


def get_giant_cc(graph, topn=10):
    """
    Return igraph.Graph instance of the giant connected component
    """
    print("Getting giant connected component..")
    connected_components = graph.clusters(mode="weak")
    subgraphs = connected_components.subgraphs()
    sizes = [i.vcount() for i in subgraphs]
    print("Top 10 biggest CCs: ", list(sizes)[:topn])
    cc_sorted = [
        subgraph
        for _, subgraph in sorted(
            zip(sizes, subgraphs), reverse=True, key=lambda pair: pair[0]
        )
    ]
    giant_cc = cc_sorted[0]
    return giant_cc


def save_edgelist(edgelist_df, fpath, sep=" "):
    # edgelist_df = get_edgelist(graph)
    _, ext = os.path.splitext(fpath)
    if ext == ".txt" or ext == ".csv":
        edgelist_df.to_csv(fpath, sep=sep, index=False, header=False)
    elif ext == ".parquet":
        edgelist_df.to_parquet(fpath, index=False)
    else:
        raise ValueError(f"Unknown file extension: {ext}")
    return


def get_edgelist(graph):
    """
    Return an edgelist df from igraph 
    NOTE: igraph.write() and write_adjacency_list() function does not save node names, so we need to write our own.
    edgelist has the following columns: ['source_name', 'target_name', 'edge ID', 'source', 'target', 'weight']
    """
    node_dict = graph.get_vertex_dataframe().reset_index()
    edge_dict = graph.get_edge_dataframe().reset_index()
    named_source = pd.merge(
        node_dict, edge_dict, left_on="vertex ID", right_on=["target"]
    )
    edgelist = pd.merge(
        node_dict, named_source, left_on="vertex ID", right_on=["source"]
    )
    edgelist = edgelist.rename(
        columns={"name_x": "source_name", "name_y": "target_name"}
    )
    edgelist = edgelist.drop(columns=["vertex ID_x", "vertex ID_y"])
    edgelist = edgelist[["source_name", "target_name", "weight"]]
    return edgelist


def summary(graph):
    average_friends = graph.ecount() / graph.vcount()
    print(
        f"{graph.vcount()} nodes and {graph.ecount()} edges"
        f"(average number of friends: {average_friends})"
    )
    return


def filter_graph(og_graph, nodes_to_filter):
    """
        - Reduce the size of network by retaining a k-core=94
        - Reduce density by delete a random sample of edges 
    """

    graph = deepcopy(og_graph)

    # Sample edges
    #  Delete a random sample of edges
    # Set the initial average in/out-degree (average_friends) as a target
    # Basic stats

    average_friends = graph.ecount() / graph.vcount()
    print(
        f"{graph.vcount()} nodes and {graph.ecount()} edges initially "
        f"(average number of friends: {average_friends})"
    )
    no_target_edges = int(graph.vcount() * average_friends)
    no_remove_edges = graph.ecount() - no_target_edges
    weights = [e["weight"] for e in graph.es]
    probs = [i / sum(weights) for i in weights]
    # get a sample of edges weighted by edge weight (WITH replacement)
    deleted_edges = np.random.choice(
        range(graph.ecount()), no_remove_edges, replace=False, p=probs
    )
    graph.delete_edges(deleted_edges)
    print("After edge sampling:")
    summary(graph)
    return graph


def extract_backbone(graph, alpha=0.05):
    """
    Return the "backbone" of the graph containing a subset of weighted edges that fall above the threshold
    Igraph implementation of 
    https://gist.github.com/brianckeegan/8846206
    NOTE: this is for undirected graph

    graph: weighted, undirected igraph
    alpha: threshold or significant level. Higher alpha means more aggressive pruning
    """

    # gist: for each node, find the significant level of of its edges by comparing
    # each edge's weight to the sum of all out-degree edge weights

    # columns: ['edge ID', 'source', 'target', 'weight']
    vertice_df = graph.get_edge_dataframe().reset_index()
    vertices = vertice_df.values

    all_significant_idxs = []
    for vertex_id in tqdm(range(graph.vcount()), desc="Extracting significant edges"):
        # equivalent to graph.neighbors(vertex_id, mode='all')
        mask = (vertices[:, 1] == vertex_id) | (vertices[:, 2] == vertex_id)
        # get edge weight to neighbors
        weights = vertices[mask, 3]
        neighbors = vertices[mask, 2].astype(int)
        # This will fail but it's ok because we only care about the weights
        # e.g: if we have source, target pairs: [0,1], [1,2], [1,3]. For vid = 1, ig_neighbors={0,2,3} but neighbors = {1,2,3}
        # ig_neighbors = graph.neighbors(vertex_id)
        # assert set(ig_neighbors) == set(neighbors)
        k_n = len(neighbors)
        if k_n > 1:
            sum_w = np.array([np.sum(weights)] * k_n)
            pij = np.divide(weights, sum_w)
            ones = np.ones(k_n)
            relevance = (ones - pij) ** (k_n - 1)
            significant_idxs = np.argwhere(relevance < alpha).flatten()  # equation 2
            all_significant_idxs.extend(significant_idxs)

    # backbone_graph_df = vertice_df[vertice_df.index.isin(all_significant_idxs)]
    edge_ids = vertice_df.loc[all_significant_idxs, "edge ID"].values
    backbone_graph = graph.subgraph_edges(edge_ids, delete_vertices=True)
    return backbone_graph


if __name__ == "__main__":
    # inpath = "data/covid/edgelists/cosharing_edgelist.txt"
    # outdir = "analyses"
    # alpha = 0.05
    direction = sys.argv[1]  # directed or undirected
    inpath = sys.argv[2]
    outdir = sys.argv[3]
    if len(sys.argv) == 5:
        alpha = float(sys.argv[4])
    else:
        alpha = None

    if direction == "undirected":
        directed = False
    elif direction == "directed":
        directed = True
    else:
        raise ValueError(
            "Ineligible value for direction of the network. {'directed', 'undirected'}"
        )
    graph = ig.Graph.Read_Ncol(inpath, directed=directed)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    subgraph = get_giant_cc(graph, topn=10)
    summary(subgraph)

    if alpha is not None:
        subgraph = extract_backbone(subgraph, alpha=alpha)

    try:
        # get subgraph edgelist
        edgelist_df = get_edgelist(subgraph)
    except Exception as e:
        print(e)
        raise ValueError("Subgraph is empty. Exting..")

    # naming
    size = subgraph.vcount()
    basename = os.path.basename(inpath)
    filename = os.path.splitext(basename)[0]
    if alpha is not None:
        file_name = f"{filename}__{size}_backbone{alpha}"
    else:
        file_name = f"{filename}__{size}"
    save_edgelist(edgelist_df, os.path.join(outdir, f"{file_name}.txt"))
    save_edgelist(edgelist_df, os.path.join(outdir, f"{file_name}.parquet"))

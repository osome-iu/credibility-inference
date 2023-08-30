"""
Purpose:
    Create node2vec embedding representation of a graph. 
    Direction is ignored if a directed network is provided
Inputs: 
    - .txt or .edge file of an egelist: tab-separated, 3 columns: (v1,v2, weight)
    - .json of configuration for the Node2vec model, namely {p, q}: parameter specifying the biased random walk
    
Outputs: 
    .pkl file of embedding (dict): {node_name: embedding_vector}
"""
import os
import networkx as nx
from infopolluter.model.graphutil import format_edgelist, save_edgelist
import sys
import argparse
import json


def elio(
    graph,
    dim: int,
    walk_length: int,
    num_walks: int,
    p: float,
    q: float,
    workers: int,
    window_size: int,
    epochs: int,
):
    # Note: don't set workers=-1
    from node2vec import Node2Vec

    # Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
    node2vec = Node2Vec(
        graph,
        dimensions=dim,
        walk_length=walk_length,
        num_walks=num_walks,
        p=p,
        q=q,
        workers=workers,
    )
    # Embed nodes
    model = node2vec.fit(window=window_size, epochs=epochs)
    # model = node2vec.fit(window=window_size, min_count=1, batch_words=4, epochs=epochs)
    return model


def main(args):
    parser = argparse.ArgumentParser(description="Get graph embedding",)
    parser.add_argument(
        "-i",
        "--infile",
        type=str,
        required=True,
        help=".txt infile path (space-separated graph edge list)",
    )

    parser.add_argument(
        "-o",
        "--outfile",
        type=str,
        required=True,
        help=".model fpath to embedding (Keyed vectors) model",
    )

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help=".json fpath for configuration to create node2vec embedding",
    )

    parser.add_argument(
        "-d",
        "--direction",
        type=str,
        required=False,
        help='Mode of embedding ("undirected", "directed", "trust")',
    )

    print(os.getcwd())
    args = parser.parse_args(args)
    edge_fp = args.infile
    config = args.config
    outfile = args.outfile
    specs = json.load(open(config, "r"))

    direction = "undirected" if args.direction is None else args.direction
    directed = True if (direction == "directed" or direction == "trust") else False
    reverse = True if direction == "trust" else False
    # TODO: handle memory issue with format_edgelist
    # edges = format_edgelist(edge_fp, outf=None, reverse=reverse)

    # if reverse is True:
    #     filename, ext = os.path.splitext(edge_fp)
    #     reverse_edge_fp = f"{filename}_reversed{ext}"
    #     save_edgelist(edges, reverse_edge_fp, sep=" ")
    #     edge_fp = reverse_edge_fp

    print("Create embedding..")

    if directed:
        elio_graph = nx.read_weighted_edgelist(edge_fp, create_using=nx.DiGraph)
    else:
        elio_graph = nx.read_weighted_edgelist(edge_fp, create_using=nx.Graph)

    # specs.pop("window_size")  # avoid illegal arg error
    emb = elio(elio_graph, **specs)

    ## Make sure dir exists, comment out if run with Snakemake
    # filename, ext = os.path.splitext(edge_fp)
    # basename = os.path.basename(filename)
    # res_dir = "results/directed"
    # if not os.path.exists(res_dir):
    #     os.makedirs(res_dir)
    # emb.save(os.path.join(res_dir, f"{args.embedding}_{basename}.model"))
    emb.save(outfile)


if __name__ == "__main__":
    main(sys.argv[1:])

    ## DEBUG LOCAL
    # edge_fp = "data/sample/rt/rt/sample_2048.txt"
    # config = "workflow/test/config.json"
    # outfile = "workflow/test/emb.model"
    # specs = json.load(open(config, "r"))

    # weighted = True
    # directed = True
    # reverse = True
    # embedding = "pecan"
    # edge_tab = edge_fp.replace(".txt", ".edge")
    # edges = format_edgelist(edge_fp, outf=edge_tab, reverse=reverse)

    # print("Create embedding..")
    # if embedding == "fast":
    #     fast_graph = fastnode2vec.Graph(edges, directed=directed, weighted=weighted)
    #     # specs = remove_illegal_kwargs(specs, fast())
    #     emb = fast(fast_graph, **specs)

    # if embedding == "elio":
    #     if directed:
    #         elio_graph = nx.read_weighted_edgelist(edge_fp, create_using=nx.DiGraph)
    #     else:
    #         elio_graph = nx.read_weighted_edgelist(edge_fp, create_using=nx.Graph)
    #     # specs = remove_illegal_kwargs(specs, elio())
    #     emb = elio(elio_graph, **specs)
    # if embedding == "pecan":
    #     # specs = remove_illegal_kwargs(specs, pecan())
    #     emb = pecan(
    #         edge_tab, verbose=False, weighted=weighted, directed=directed, **specs
    #     )

    # emb.save(outfile)
    # print("")

from fastnode2vec import Graph, Node2Vec
import argparse
from infopolluter import graphutil
from gensim.models.keyedvectors import KeyedVectors
import os
import sys


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--edgelist",
        type=str,
        help="File path to edgelist (.txt)",
    )

    parser.add_argument(
        "-o",
        "--outpath",
        type=str,
        help="Path to save w2v model",
    )
    parser.add_argument(
        "--directed",
        type=str,
        required=False,
        help="Bool for directed graph or not",
    )

    # parser.add_argument(
    #     "--config",
    #     action="store",
    #     dest="config",
    #     type=str,
    #     required=False,
    #     help="path to config file",
    # )

    args = parser.parse_args(args)
    directed = False if args.directed is None else True
    # params = json.load(open(args.config, "r"))
    print(f"Generating node2vec embeddings (directed={directed}).. ")
    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)
    try:
        net = graphutil.format_edgelist(
            inf=args.edgelist, outf=None, insep=" ", outsep="\t"
        )
        graph = Graph(net, directed=directed, weighted=True)
        n2v = Node2Vec(
            graph, dim=128, walk_length=80, context=10, p=0.25, q=0.1, workers=-1
        )

        n2v.train(epochs=50)

        print("Done. Saving..")
        filename, file_extension = os.path.splitext(args.edgelist)
        fname = os.path.basename(filename)
        wv_out = os.path.join(args.outpath, f"fastnode2vec_{fname}.kv")
        n2v.wv.save(wv_out)

        # Test saved model
        reloaded_word_vectors = KeyedVectors.load(wv_out)
        nodename = net[0][0]
        dim = len(reloaded_word_vectors[nodename])
        print(f"Test saved model.. Vector dimension: {dim}")
        print("Finished!")
    except Exception as e:
        print(e)
        pass


if __name__ == "__main__":
    main(sys.argv[1:])

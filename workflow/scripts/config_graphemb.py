import json
import argparse
import sys


def main(args):
    parser = argparse.ArgumentParser(
        description="Make node2vec config",
    )
    parser.add_argument(
        "-p",
        type=float,
        required=True,
        help="1/p: likelihood of revisiting nodes, keeps the walk local",
    )

    parser.add_argument(
        "-q",
        type=float,
        required=True,
        help="1/q: likelihood of exploration, inclined to visit nodes further away",
    )

    parser.add_argument(
        "-o",
        "--outfile",
        type=str,
        required=True,
        help=".json fpath to output config",
    )

    args = parser.parse_args(args)
    config = {
        "walk_length": 80,
        "num_walks": 10,
        "dim": 128,
        "p": args.p,
        "q": args.q,
        "workers": 4,
        "window_size": 10,
        "epochs": 5,
    }

    # save config
    json.dump(config, open(args.outfile, "w"))


if __name__ == "__main__":
    main(sys.argv[1:])

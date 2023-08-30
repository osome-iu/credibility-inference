from infopolluter import Population, Classifier
from infopolluter import summarize_evaluation
from infopolluter import get_text2vec_mapping, load_glove_embedding
from infopolluter.util import get_account_labels
import pandas as pd
import os
import pickle
import argparse
import json
import sys


def run_evaluation(label_path, edgelist, embedding_path, data_path, eval_mode="preds"):
    """
    - extract embedding features by mapping user's tweets to glove embedding
    - add these features to population using pop.add_metadata
    - train an infopolluter classifier on these graph features from the training corpus.
    We then use the trained classifier to compute the ranking for users in the test corpus
    """
    # Format labels
    label_dict = get_account_labels(label_path)

    # Prep users
    pop = Population(edgefilename=edgelist)
    pop.add_labels(label_dict)

    # Add features
    print("Load embedding..")
    w2v_model = load_glove_embedding(embedding_path)
    df = pd.read_parquet(data_path)
    embeddings = get_text2vec_mapping(df, w2v_model)
    pop.add_user_metadata("glove", embeddings)

    print("Classification..")
    # Do classification
    clf = Classifier(
        obj_type="user",
        pred_feats=["glove"],
        labeller=lambda user: user.meta["label"] == 1,
    )
    known_pop = pop.filter_users_by(selector=lambda user: user.meta["label"] != None)
    print(f"pop with known label: {len(known_pop.users)}")
    # report = clf.evaluate_with_cv(corpus=known_pop, verbose=True)

    if eval_mode == "preds":
        true, preds, uids = clf.predict_with_cv(corpus=known_pop)
        report = summarize_evaluation(true, preds, find_optimal=True)
        df = pd.DataFrame(data={"uid": uids, "y_true": true, "y_pred": preds})
    else:
        # Average metrics to explore params
        report = clf.evaluate_with_cv(corpus=known_pop, verbose=True)
        df = pd.DataFrame(columns=["dummy_col"])
    return df, report


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--edgelist", type=str, help="File path to edgelist (.txt)",
    )
    parser.add_argument(
        "-e",
        "--w2vfile",
        type=str,
        help="File path for .pkl file of graph embedding (dict of node-vector)",
    )
    parser.add_argument(
        "-t",
        "--textfile",
        type=str,
        help="File path for .parquet file of user and associated posts",
    )

    parser.add_argument(
        "-l", "--labelfile", type=str, help="File path for df of user labels"
    )

    parser.add_argument(
        "-o",
        "--outfile",
        type=str,
        help="Path to save results (raw predictions and AUC)",
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=False,
        action="store",
        help="Mode of evaluation (if 'preds': save predictions, 'cv': return cv scores)",
    )

    args = parser.parse_args(args)
    # params = json.load(open(args.config, "r"))
    print("Start evaluating GloVe classification.. ")
    try:
        df, report = run_evaluation(
            label_path=args.labelfile,
            edgelist=args.edgelist,
            embedding_path=args.w2vfile,
            data_path=args.textfile,
            eval_mode=args.mode,
        )
        # filename, file_extension = os.path.splitext(args.edgelist)

        print("Done. Saving..")

        df_out = args.outfile
        report_out = args.outfile.replace(".parquet", ".pkl")
        pickle.dump(report, open(report_out, "wb"))
        df.to_parquet(df_out, engine="pyarrow")

        # fname = os.path.basename(filename)
        # df_out = os.path.join(args.outpath, f"{fname}.parquet")
        # report_out = os.path.join(args.outpath, f"{fname}.pkl")
        # pickle.dump(report, open(report_out, "wb"))
        # df.to_parquet(df_out, engine="pyarrow")

    except Exception as e:
        print(e)
        pass


if __name__ == "__main__":
    main(sys.argv[1:])

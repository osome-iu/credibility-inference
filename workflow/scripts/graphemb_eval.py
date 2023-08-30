from infopolluter import Population, Classifier
from infopolluter import summarize_evaluation
from infopolluter import word2vec_object_to_dict
from infopolluter.util import get_account_labels
from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Word2Vec
import pandas as pd
import os
import pickle
import argparse
import sys


def run_evaluation(label_path, edgelist, embedding_path, eval_mode="eval"):
    """
    - add graph embedding features (node2vec) to population using pop.add_metadata
    - train an infopolluter classifier on these graph features from the training corpus
    We then use the trained classifier to compute the ranking for users in the test corpus
    Inputs: 
        - label_path: path to label file. Has columns: user_id, label
        - edgelist: path to edgelist (space-separated .txt files or .parquet)
        - embedding_path: path to embedding file. Expects .pkl of dict(user_id: node2vec vector) or .model file of Word2Vec model
        - eval_mode: {preds, eval}. Default is "eval": only returning metrics. "preds" returns predictions
    Return a report and a df of predidtions
    """
    # Format labels
    label_dict = get_account_labels(label_path)
    # Prep users
    pop = Population(edgefilename=edgelist)
    pop.add_labels(label_dict)

    # Add features
    filename, ext = os.path.splitext(embedding_path)
    if ext == ".pickle" or ext == ".pkl":
        w2v_model = pickle.load(open(embedding_path, "rb"))
    else:
        w2v_model = KeyedVectors.load(embedding_path)
    if isinstance(w2v_model, KeyedVectors) or isinstance(w2v_model, Word2Vec):
        embeddings = word2vec_object_to_dict(w2v_model)
    else:
        # expect dict of user-node2vec vectors
        embeddings = w2v_model
    pop.add_user_metadata("node2vec", embeddings)

    # Do classification
    clf = Classifier(
        obj_type="user",
        pred_feats=["node2vec"],
        labeller=lambda user: user.meta["label"] == 1,
    )
    known_pop = pop.filter_users_by(selector=lambda user: user.meta["label"] != None)
    print(f"Population with known label: {len(known_pop.users)}")

    if eval_mode == "preds":
        true, preds, uids = clf.predict_with_cv(corpus=known_pop)
        report = summarize_evaluation(true, preds, find_optimal=True)
        df = pd.DataFrame(data={"uid": uids, "y_true": true, "y_pred": preds,})
    else:
        # Average metrics to explore params
        report = clf.evaluate_with_cv(corpus=known_pop, verbose=True)
        df = pd.DataFrame(columns=["dummy_col"])
        # NOTE: predict_with_cv and evaluate_with_cv() don't return a corpus, so they don't work with summarize
        # df = clf.summarize(known_pop)
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
        "-l", "--labelfile", type=str, help="File path for df of user labels"
    )
    parser.add_argument(
        "-o",
        "--outfile",
        type=str,
        help=".parquet path to save results (raw predictions and AUC)",
    )
    parser.add_argument(
        "--cv", type=int, help="Number of folds for cross-validation",
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
    print("Start evaluating Node2vec classification.. ")
    dir_name = os.path.dirname(args.outfile)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    try:
        df, report = run_evaluation(
            args.labelfile, args.edgelist, args.w2vfile, eval_mode=args.mode
        )
        # filename, file_extension = os.path.splitext(args.w2vfile)

        # print("Done. Saving..")
        # fname = os.path.basename(filename)
        # df_out = os.path.join(args.outpath, f"{fname}.parquet")
        # report_out = os.path.join(args.outpath, f"{fname}.pkl")
        df_out = args.outfile

        report_out = args.outfile.replace(".parquet", ".pkl")
        pickle.dump(report, open(report_out, "wb"))
        df.to_parquet(df_out, engine="pyarrow")

        # pickle.dump(report, open(args.outfile))
        # df.to_parquet(args.verboseoutfile, engine="pyarrow")
    except Exception as e:
        print(e)
        pass


if __name__ == "__main__":
    main(sys.argv[1:])

    # # DEBUG
    # edgelist = "data/covid/edgelists/cosharing_edgelist__67539.txt"
    # labelfile = "data/user_labels.csv"
    # w2vfile = "analyses/cosharing_viz/data/pca_scalingtrue.pkl"
    # # w2vfile = "data/covid/node2vec/fast_undirected_p2q1__cosharing_67539.model"
    # df, report = run_evaluation(labelfile, edgelist, w2vfile, eval_mode="")


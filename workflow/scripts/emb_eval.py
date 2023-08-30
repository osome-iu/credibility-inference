"""
Run cross_validate on classifier using graph embedding (or text embedding) as feature 
"""
from infopolluter import PopulationClassifier, Population
from infopolluter import optimal_threshold, summarize_evaluation
from sklearn import metrics
import numpy as np
from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    StratifiedKFold,
)

from infopolluter import word2vec_object_to_dict
from gensim.models.keyedvectors import KeyedVectors
import pandas as pd
import pickle
import os
from collections import defaultdict
import argparse
import sys
from infopolluter.util import user_col, threshold_label
import collections

RANDOM_STATE = 42


def process_results(results, test_idx, prediction_name="label"):
    """
    results (Population): data return from PopulationClassifier.predict()
    prediction_name: metadata field name of prediction score returned by PopulationClassifier: {label, probability}
    Return y_pred, y_true arrays with the same dimension
    """
    y_pred = []
    y_true = []

    # NOTE: prediction only available for users whose confidence is 100
    for user in results.iter_users(
        selector=lambda user: user.meta["user_index"] in test_idx
    ):
        ground = user.get_meta("true_label")
        y_true.append(ground)
        # y_pred.append(user.meta[clf.clf_prob_attribute_name])
        y_pred.append(user.meta[prediction_name])

    print(f"Evaluating..: {len(y_pred)} users")

    if set(y_pred) != {1, 0}:
        # not binary classification, need to find optimal threshold
        y_pred = optimal_threshold(y_true, y_pred, pos_label=1)
    return y_pred, y_true


def confusion_matrix_scorer(clf, X, y):
    # b: X,y here is the test set, equivalent to (X_test, y_test) in train_test_split
    # cross_validate() has called clf.init() and clf.fit(X_train, y_train) before this
    print("** Scorer.. ")
    print(f"Len X: {len(X)}")
    pred_pop = clf.predict(X)
    y_pred, y_true = process_results(
        pred_pop, X, prediction_name=clf.clf_attribute_name
    )
    report = summarize_evaluation(y_true, y_pred, pos_label=1)
    return report


def eval_with_train_test_split(clf, X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    clf.fit(X_train, y_train)
    pred_pop = clf.predict(X_test)
    y_pred, y_true = process_results(
        pred_pop, X_test, prediction_name=clf.clf_attribute_name
    )
    report = summarize_evaluation(y_true, y_pred, pos_label=1)

    print(
        f"\nF1 (binary class):{np.round(report['f1'], 3)}"
        f"\nF1 (macro):{np.round(report['f1_macro'], 3)}"
        f"\nAUC: {np.round(report['auc'], 3)}"
    )
    return report


def eval_with_cv(clf, X, y, n_splits=5, random_state=RANDOM_STATE):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    report = collections.defaultdict(list)
    for i, (train_index, test_index) in enumerate(cv.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(X_train, y_train)
        pred_pop = clf.predict(X_test)
        y_pred, y_true = process_results(
            pred_pop, X_test, prediction_name=clf.clf_attribute_name
        )
        report_i = summarize_evaluation(y_true, y_pred, pos_label=1)
        for key in report_i.keys():
            report[key].append(report_i[key])

    print(
        f"Average results across {n_splits} folds:"
        f"\nF1 (binary class):{np.round(np.mean(report['f1']), 3)} pm {np.round(np.std(report['f1']), 3)}"
        f"\nF1 (macro):{np.round(np.mean(report['f1_macro']), 3)} pm {np.round(np.std(report['f1_macro']), 3)}"
        f"\nAUC: {np.round(np.mean(report['auc']), 4)} pm {np.round(np.std(report['auc']), 3)}"
    )
    return dict(report)


def prep_data(
    edgelist, embedding_path, user_labels, confidence_level=1, predictor="knn"
):
    """
    Return X,y
    Input:
        - edgelist
        - embedding_path
        - user_labels
    """
    population = Population(edgefilename=edgelist)
    population_users = [user.id for user in population.iter_users()]

    user_info = pd.read_parquet(user_labels)
    user_info = user_info[user_info[user_col].isin(population_users)]
    user_info["user_index"] = user_info.index
    user_info["true_label"] = user_info["user_score"].apply(
        lambda x: threshold_label(x)
    )
    idxs = user_info.set_index(user_col)["user_index"].to_dict()
    true_labels = user_info.set_index(user_col)["true_label"].to_dict()
    known_frac = user_info.set_index(user_col)["confidence"].to_dict()

    # Add labels
    population.add_user_metadata("user_index", idxs)
    population.add_user_metadata("true_label", true_labels)
    population.add_user_metadata("confidence", known_frac, default=0)
    # Add features
    filename, ext = os.path.splitext(embedding_path)
    if ext == ".pickle" or ext == ".pkl":
        w2v_model = pickle.load(open(embedding_path, "rb"))
    else:
        w2v_model = KeyedVectors.load(embedding_path)

    embeddings = word2vec_object_to_dict(w2v_model)
    population.add_user_metadata("node2vec", embeddings)

    clf = PopulationClassifier(
        population=population,
        pred_feats=["node2vec"],
        labeller=None,
        predictor=predictor,
    )
    ## DO EVALUATION
    user_labels = user_info[["user_index", "confidence"]].values
    # user_labels = user_info[["domain_index", "Score", "label"]].values
    known_idx = np.argwhere(user_labels[:, 1] >= confidence_level)
    print(
        f"Evaluating on accounts with confidence >= {confidence_level}: {len(known_idx)}"
    )
    known = user_labels[known_idx.ravel(), :].astype(np.int32)

    X = known[:, 0]
    y = known[:, 1]

    return clf, X, y


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--edgelist", type=str, help="File path to edgelist (.txt)",
    )
    parser.add_argument(
        "-e",
        "--w2vfile",
        type=str,
        help=".pkl file of graph embedding (dict of node-vector)",
    )
    parser.add_argument(
        "-l", "--userlabels", type=str, help=".parquet file path to df of user labels",
    )
    parser.add_argument(
        "-o", "--outfile", type=str, help=".pkl path to save evaluation report",
    )

    parser.add_argument(
        "--predictor", type=str, help="predictor {dectree, randforest, logreg, knn}",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        help="Confidence threshold to subset data for evaluation",
    )

    parser.add_argument(
        "--cv", type=int, help="Number of folds for cross-validation (default: 5)",
    )
    args = parser.parse_args(args)
    edgelist = args.edgelist
    embedding_path = args.w2vfile
    user_labels = args.userlabels
    predictor = args.predictor
    confidence = args.confidence
    no_splits = args.cv if args.cv is not None else 5

    outpath = os.path.dirname(args.outfile)
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    print(
        f"Node2vec classification.. ({no_splits} folds CV -- confidence level:{confidence})"
    )
    try:
        clf, X, y = prep_data(
            edgelist,
            embedding_path,
            user_labels,
            confidence_level=confidence,
            predictor=predictor,
        )
        report = eval_with_cv(clf, X, y, n_splits=no_splits)
        pickle.dump(report, open(args.outfile, "wb"))
    except Exception as e:
        print(e)
        pass


if __name__ == "__main__":
    main(sys.argv[1:])
    # LOCAL DEBUG
    # edgelist = "data/exp/rt_cc_size322208.txt"
    # embedding_path = "data/exp/cc_size322208_node2vec3.pkl"
    # # embedding_path = "data/covid/derived/undirected/elio_p1q0.5__rt_cc_size322208.model"
    # user_labels = "data/covid/labels/user_info.parquet"

    # clf, X, y = prep_data(edgelist, embedding_path, user_labels)
    # report = eval_with_train_test_split(clf, X, y)

    # report = eval_with_cv(clf, X, y, n_splits=CV_FOLDS)
    # pickle.dump(report, open("covaxxy_emb_eval.pkl", "wb"))

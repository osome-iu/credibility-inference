from infopolluter import CentralityRanker
from infopolluter import summarize_evaluation
from sklearn.model_selection import cross_validate
import numpy as np
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    cross_validate,
    KFold,
    StratifiedKFold,
)

import pandas as pd
import pickle
import argparse
import sys
from infopolluter.util import threshold_label, user_col
from infopolluter import graph_from_file
import collections

RANDOM_STATE = 42
POSITIVE_LABEL = 1


def eval_with_cv(ranker, X, y, n_splits=5):
    # POSITIVE_LABEL = 1 if ranker.method == "locred" else 0
    report = collections.defaultdict(list)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    for i, (train_index, test_index) in enumerate(cv.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        ranker.fit(X_train, y_train)
        results = ranker.predict(X_test)
        y_true = results.y_true
        y_pred = results.y_pred
        report_i = summarize_evaluation(
            y_true, y_pred, pos_label=POSITIVE_LABEL, find_optimal=True
        )
        if report_i is not None:
            for key in report_i.keys():
                report[key].append(report_i[key])

    print(
        f"Average results across {n_splits} folds:"
        f"AUC:{np.round(np.mean(report['auc']), 3)} pm {np.round(np.std(report['auc']), 3)}"
        f"\nF1 (binary class):{np.round(np.mean(report['f1']), 3)} pm {np.round(np.std(report['f1']), 3)}"
        f"\nF1 (macro):{np.round(np.mean(report['f1_macro']), 3)} pm {np.round(np.std(report['f1_macro']), 3)}"
    )

    return report


def eval_with_train_test_split(
    ranker, X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=True
):
    if stratify == True:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    ranker.fit(X_train, y_train)
    results = ranker.predict(X_test)
    y_true = results.y_true
    y_pred = results.y_pred

    report = summarize_evaluation(
        y_true, y_pred, pos_label=POSITIVE_LABEL, find_optimal=True
    )
    print(
        f"AUC:{np.round(report['auc'], 3)}"
        f"\nF1 (binary class):{np.round(report['f1'], 3)}"
        f"\nF1 (macro):{np.round(report['f1_macro'], 3)}"
    )

    return report


def confusion_matrix_scorer(clf, X, y):
    # TODO: Change predict() to return population
    results = clf.predict(X)

    y_true = results.y_true
    y_pred = results.y_pred

    report = summarize_evaluation(
        y_true, y_pred, pos_label=POSITIVE_LABEL, find_optimal=True
    )
    return report


def prep_data(
    label_fpath,
    edgelist,
    method="locred",
    confidence_level=1,
    alpha=0.8,
    vary_alpha=False,
):
    """
    Prepare data for evaluation.
    Return X,y and CentralityRanker instance
    Inputs:
    - edgelist: File path to edgelist (.txt, .graphml or .parquet)
    - label_fpath: File path to df of user labels. ["user_id", "user_score", "confidence"]
    - method: Method to use for ranking
    - confidence_level: Confidence threshold
    """

    # Labels
    user_info = pd.read_parquet(label_fpath)
    user_info = user_info[user_info["user_score"] != -1].reset_index()
    user_info["true_label"] = user_info["user_score"].apply(
        lambda x: threshold_label(x)
    )
    user_info = user_info[["user_id", "confidence", "true_label"]]

    # Graph
    graph, vertex_df = graph_from_file(edgelist)
    # users in graph who are labeled
    labeled_vertex = pd.merge(
        vertex_df, user_info, left_on="name", right_on=user_col, how="left",
    )
    assert set(labeled_vertex["name"]) == set(vertex_df["name"])
    # drop user_col from right table to avoid confusion
    labeled_vertex = labeled_vertex.drop(user_col, axis=1)
    labeled_vertex[user_col] = labeled_vertex["name"]

    # Ranker
    ranker = CentralityRanker(
        graph=graph,
        vertex_df=vertex_df,
        user_info=labeled_vertex,
        method=method,
        alpha=alpha,
        vary_alpha=vary_alpha,
    )
    known = labeled_vertex[labeled_vertex["confidence"] >= confidence_level][
        ["vertex ID", "confidence", "true_label"]
    ].values.astype(np.int64)

    # TODO: Check the overlap between users labeled in predict() and users in evaluation
    # X contains vertex ID and their true labels
    X = known[:, [0, 2]]
    y = known[:, 2]
    return X, y, ranker


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--edgelist",
        type=str,
        required=True,
        help="File path to edgelist (.txt, .graphml or .parquet)",
    )

    parser.add_argument(
        "-l",
        "--userlabels",
        type=str,
        required=True,
        help='.parquet file path to df of user labels. ["user_id", "user_score", "confidence"]',
    )
    parser.add_argument(
        "-o",
        "--outfile",
        type=str,
        help=".parquet path to save results (raw predictions and AUC)",
    )

    parser.add_argument(
        "--method",
        type=str,
        help="Centrality method to use: {locred, reputation_scaling, ppt, pt, trustrank}",
    )

    parser.add_argument(
        "-a", "--alpha", type=float, help="Value for alpha (jumping factor)",
    )

    parser.add_argument(
        "--varyalpha",
        action="store_true",
        help="(Apply to Reputation scaling and TrustRank) If True, use different values of alphas: alpha2=(1-alpha1) ",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        help="Confidence threshold to subset data for evaluation",
    )

    args = parser.parse_args(args)

    outfile = args.outfile

    ## Initialization
    X, y, ranker = prep_data(
        label_fpath=args.userlabels,
        edgelist=args.edgelist,
        method=args.method,
        alpha=args.alpha,
        vary_alpha=args.varyalpha,
        confidence_level=args.confidence,
    )

    # cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    # cv_results = cross_validate(
    #     ranker, X=X, y=y, cv=cv, scoring=confusion_matrix_scorer
    # )
    # pickle.dump(cv_results, open(outfile, "wb"))
    report = eval_with_cv(ranker, X, y)
    pickle.dump(report, open(outfile, "wb"))


if __name__ == "__main__":
    main(sys.argv[1:])

    # # LOCAL DEBUG
    # CONFIDENCE = 1
    # method = "locred"
    # # edgelist = "data/covid/edgelists/rt_edgelist.txt"
    # # edgelist = "/Users/baott/infopolluters/data/exp/rt_cc_size322208.txt"
    # edgelist = "/Users/baott/infopolluters/data/covid/edgelists/rt_edgelist.txt"
    # label_fpath = "data/covid/labels/user_info.parquet"
    # outfile="test.pkl"
    # X, y, ranker = prep_data(
    #     label_fpath=label_fpath,
    #     edgelist=edgelist,
    #     method="locred"
    # )
    # report = eval_with_cv(ranker, X, y)
    # pickle.dump(report, open(outfile, "wb"))

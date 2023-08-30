"""
Script to evaluate performance of Centrality ranking algorithms on a bipartite networks 
"""
from infopolluter import summarize_evaluation
from infopolluter.util import threshold_label
from infopolluter import BipartiteNetwork
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
)
import numpy as np
import collections
import argparse
import sys
import pickle

DOMAIN_COL = "domain"
USER_COL = "user_id"
WEIGHT_COL = "weight"
LABEL_COL = "label"
POSITIVE_LABEL = 1
RANDOM_SEED = 42


def cocred_predict(df, true_labels, normalizer="CoCred", **kwargs):
    """
    Return df of predictions on bipartite network
    Input:
        - df: bipartite edgelist where top_col and bottom_col are nodes, and weight_col is edge weight,
        and label_col contains dummy domain labels
        - true_labels: df of actual labels for a subset of users (bottom_col)
    Output:
        - df of predictions: at least 3 columns: USER_COL, y_true, y_pred
    """
    bn = BipartiteNetwork()
    bn.set_edgelist(df, **kwargs)
    dscores, uscores = bn.generate_birank(normalizer=normalizer)

    ## Merge with y true:
    results = pd.merge(true_labels, uscores, on=USER_COL, how="left")
    results = results.rename(
        columns={"true_label": "y_true", f"{USER_COL}_birank": "y_pred"}
    )

    return results


def get_report(
    cocred_df,
    eval_users,
    normalizer="CoCred",
    top_col=DOMAIN_COL,
    bottom_col=USER_COL,
    weight_col=WEIGHT_COL,
    label_col="dummy_user",
):
    """
    Run Cocred 
    Return classification report
    cocred_df: bipartite edgelist with 2 extra columns: user_label
    eval_users: df of users to evaluate (subset of labeled users whose confidence==1). Cols: ["user_id", "true_label"] 
    """
    # hide the users from bipartite edgelist:
    test_users = list(eval_users[USER_COL].values)
    cocred_df[label_col] = [-1] * len(cocred_df)

    fit_idx = cocred_df[~cocred_df[USER_COL].isin(test_users)].index
    hide_idx = cocred_df[cocred_df[USER_COL].isin(test_users)].index
    cocred_df.loc[fit_idx, label_col] = cocred_df.loc[fit_idx, "true_label"]
    nans = np.empty(len(hide_idx))
    nans.fill(-1)
    hidden = cocred_df.loc[hide_idx, label_col].values
    # make sure all labels for users in evaluation are hidden
    assert (hidden == nans).all()

    # get predictions
    results = cocred_predict(
        cocred_df,
        eval_users,
        normalizer=normalizer,
        top_col=top_col,
        bottom_col=bottom_col,
        weight_col=weight_col,
        label_col=label_col,
    )
    y_true = results.y_true
    y_pred = results.y_pred
    report = summarize_evaluation(
        y_true, y_pred, pos_label=POSITIVE_LABEL, find_optimal=True
    )
    return report


def prep_data(edgelist, user_info, confidence_level=1):
    """
    - Label bipartite edgelist 
    - Filter edgelist & label df for users with confidence >= threshold
    Return: 
    - cocred_df: bipartite edgelist with 1 extra column: true_label (binary labels for users)
    - known: label df of users with confidence >= threshold
    """
    # Prep data
    df = pd.merge(edgelist, user_info, on=USER_COL, how="left")
    df["true_label"] = df["user_score"].apply(lambda x: threshold_label(x))

    cocred_df = df[df["confidence"] >= confidence_level][
        [USER_COL, DOMAIN_COL, WEIGHT_COL, "true_label"]
    ].reset_index(drop=True)

    known = (
        df[df["confidence"] >= confidence_level]
        .drop_duplicates(USER_COL)
        .reset_index(drop=False)
    )
    print(f"Number of users with confidence>={confidence_level}: {len(known)}")

    return known, cocred_df


def evaluate_with_cv(
    edgelist,
    user_info,
    confidence_level=1,
    stratified: bool = True,
    n_splits: int = 5,
    normalizer="CoCred",
    random_state=RANDOM_SEED,
):
    """
    Input:
        df: bipartite edgelist with 2 extra columns: domain_label and user_label
    """
    ## PREP DATA
    known, cocred_df = prep_data(edgelist, user_info, confidence_level=confidence_level)
    X = known
    y = known["true_label"].values

    # Split data for evaluation
    if stratified == True:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    report = collections.defaultdict(list)

    for _, test_index in cv.split(X, y):
        # In theory test_index== known.index
        idxs = known.loc[test_index].index
        assert (test_index == idxs).all()
        # NOTE: test_index is a list of indices RELATIVE TO known (range 0-len(known))
        # Get ground truth labels for hidden users
        eval_users = known.loc[test_index, [USER_COL, "true_label"]].reset_index(
            drop=True
        )
        report_i = get_report(cocred_df, eval_users, normalizer=normalizer)

        for key in report_i.keys():
            report[key].append(report_i[key])

    print(
        f"Average results across {n_splits} folds:"
        f"\nF1 (binary class):{np.round(np.mean(report['f1']), 3)} pm {np.round(np.std(report['f1']), 3)}"
        f"\nF1 (macro):{np.round(np.mean(report['f1_macro']), 3)} pm {np.round(np.std(report['f1_macro']), 3)}"
        f"\nAUC: {np.round(np.mean(report['auc']), 4)} pm {np.round(np.std(report['auc']), 3)}"
    )
    return dict(report)


def evaluate_with_train_test_split(
    edgelist, user_info, confidence_level=1, test_size=0.2, normalizer="CoCred"
):
    """
    Return a report of performance evaluation on a testing set of users.
        
        - confidence_level: Only use information of users with a confidence level above this threshold to initialize the propagation. Range: [0, 1]
        - test_size=0.2
    """
    ## PREP DATA
    known, cocred_df = prep_data(edgelist, user_info, confidence_level=confidence_level)

    ## TRAIN_TEST SPLIT
    X = known.index
    y = known["true_label"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    # X_test is the index of test samples. NOTE: this index the known users & doesn't match the indexed bipartite graph `cocred_df` .
    # Get ground truth labels for hidden users
    eval_users = known.loc[X_test, [USER_COL, "true_label", "confidence"]].reset_index(
        drop=True
    )
    assert len(eval_users) == len(X_test)
    report = get_report(cocred_df, eval_users, normalizer=normalizer)

    print(
        f"\nF1 (binary class):{np.round(report['f1'], 3)}"
        f"\nF1 (macro):{np.round(report['f1_macro'], 3)}"
        f"\nAUC: {np.round(report['auc'], 3)}"
    )
    return report


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--edgelist", type=str, help="File path to edgelist (.parquet)",
    )

    parser.add_argument(
        "-u", "--userlabel", type=str, help="File path for df of user labels"
    )

    parser.add_argument(
        "-o", "--outfile", type=str, help=".pkl path to save report",
    )

    parser.add_argument(
        "--method",
        type=str,
        help='Bipartite centrality method to use: {"CoCred", "HITS", "CoHITS", "BGRM", "BiRank")',
    )

    parser.add_argument(
        "-a", "--alpha", type=float, help="Value for alpha (jumping factor)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        help="Confidence threshold to subset data for evaluation",
    )
    args = parser.parse_args(args)

    try:
        edgelist = pd.read_parquet(args.edgelist)
        user_info = pd.read_parquet(args.userlabel)
        report = evaluate_with_cv(
            edgelist,
            user_info,
            normalizer=args.method,
            confidence_level=args.confidence,
        )
        with open(args.outfile, "wb") as f:
            pickle.dump(report, f)
    except Exception as e:
        print(e)
        pass


if __name__ == "__main__":
    main(sys.argv[1:])
    # ## DEBUG
    # methods = ["HITS", "BGRM", "BiRank"]
    # edgelist_path = (
    #     "/Users/baott/infopolluters/data/covid/edgelists/bipartite_edgelist.parquet"
    # )
    # user_info_path = "data/covid/labels/user_info.parquet"
    # edgelist = pd.read_parquet(edgelist_path)
    # user_info = pd.read_parquet(user_info_path)
    # report = evaluate_with_cv(edgelist, user_info, normalizer="HITS")

    # report = evaluate_with_train_test_split(
    #     edgelist, user_info, test_size=0.7, normalizer=method
    # )
    # for method in methods:
    #     report = evaluate_with_cv(edgelist, user_info, normalizer=method)


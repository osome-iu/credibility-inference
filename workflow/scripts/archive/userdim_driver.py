""" Classify users using 4 dimensions of trust 
    Use code of graphemb_driver but initialize embeddings with empty list
"""

import infopolluter.GraphEmb as graphemb

# import infopolluter.GraphEmb_KNN as graphemb
import infopolluter.utils as utils
import infopolluter.ip_utils as iputils
import pickle as pkl
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import sys
import argparse
import json
import copy
import os
import igraph

label_col = "label"
score_col = "score"
user_col = "uid"
name_map = {"true_label": label_col, "mean_score_times": score_col, "uid": user_col}


def rank_with_hidden_labels_locred(
    ranker,
    test_idx,
    rtedgelist,
    trustedgelist,
    score_type="mistrust",
    complement=False,
    alpha=0.85,
):
    """
    Use test_idx to hide some labels and perform ranking
    Input: GraphEmb object initialized with all labels
    test_idx:
    Return ranking for only the test_idx nodes
    options:
    [intrinsic_bad, mistrust, trust, combination, all]
    """
    # hide test labels, only feed labels that are in y_train
    # TODO: When hiding labels, we should focus on bad or good nodes only depending on personalization
    # e.g: with bad personalization, good labels is effectively the same thing as no label

    new_labels = ranker.node_labels.copy(deep=True)
    new_labels.loc[test_idx, "label"] = len(test_idx) * [np.nan]

    ranker.add_features(
        rtedge_fpath=rtedgelist,
        trustedge_fpath=trustedgelist,
        labels=new_labels,
        score_type=score_type,
        complement=complement,
        weight_col="weight",
        alpha=alpha,
    )

    all_preds = graphemb.get_ranking(ranker.features, new_labels)
    y_pred = np.array(all_preds)[test_idx]

    return y_pred


def cross_eval_graphemb(
    labels,
    rtedgelist=None,
    trustedgelist=None,
    score_type=None,
    complement=False,
    alpha=0.85,
    kfold=5,
    train_size=0.8,
    random_state=42,
    undersample=True,
):
    # TODO: add option of embedding params
    """
    Labels: df with columns: ['label', 'uid']
    params: p=param['p'], q=param['q'], workers=1, verbose=True
    num_walks=10, walk_length=80, n_ckpts=10, pb_len=100
    walks,vector_size=128, window=5, min_count=0, sg=1, workers=3, epochs=5
    """
    # TODO: clean up!
    print("Train_size:", train_size)
    ranker = graphemb.GraphEmb()

    graph = igraph.Graph.Read_Ncol(rtedgelist)
    nodes = [i["name"] for i in graph.vs]
    embeddings = {node: [] for node in nodes}
    # Initialize ranker with empty list for embeddings!
    ranker.set_embeddings(embeddings)
    ranker.add_labels(labels, undersample=undersample)

    labeled_df = ranker.get_known_labels()

    # X: df, Y: df of labels (index matching ranker.embedding_df)
    X = labeled_df.loc[
        :, ["uid"]
    ]  # what X is doesn't really matter, we just want index preserved.
    Y = labeled_df[label_col]

    traintest_sets = []  # list of (trainidx, testidx) split from X,Y
    report = defaultdict(lambda: [])

    if kfold == 1:
        # train_test_split returns arrays of train & test
        # b: this returns the whole train, test data. y_test index is preserved so it's ok
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, train_size=train_size, random_state=random_state
        )
        trainidx = sorted(y_train.index)
        testidx = sorted(y_test.index)
        traintest_sets += [(trainidx, testidx)]

    elif kfold >= 2:
        # cv = StratifiedKFold(n_splits=kfold, random_state=random_state, shuffle=True)
        cv = StratifiedShuffleSplit(
            n_splits=kfold, train_size=train_size, random_state=random_state
        )
        kfold_splits = cv.split(
            X, Y
        )  # Provides train/test indices to split data in train/test sets.
        # !!!b: traintest idx here is relative to X,Y , so we need X, Y index to match that of ranker.embedding_df
        traintest_sets = [
            (sorted(np.array(X.index)[trainidx]), sorted(np.array(Y.index)[testidx]))
            for trainidx, testidx in kfold_splits
        ]

    # We don't really use train data in this evaluation.
    # (The set of known labels are changed each time, while the embeddings of nodes remain the same)
    for (
        trainidx,
        testidx,
    ) in (
        traintest_sets
    ):  # Provides train/test indices to split data in train/test sets.
        try:
            y_pred = rank_with_hidden_labels_locred(
                ranker,
                testidx,
                rtedgelist,
                trustedgelist,
                score_type=score_type,
                alpha=alpha,
                complement=complement,
            )

            pos_label = 1
            neg_label = 0
            # pos_label=0 if score_type=='trust' else 1
            # neg_label=1 if score_type=='trust' else 0

            y_true = Y[testidx]

            report["train_no"] += [len(trainidx)]
            report["test_no"] += [len(testidx)]
            ratio = Counter(y_true)
            report["test_negative_no"] += [ratio[neg_label]]
            report["test_positive_no"] += [ratio[pos_label]]
            assert len(y_true) == len(y_pred)
            fpr, tpr, thresholds = metrics.roc_curve(
                y_true, y_pred, pos_label=pos_label
            )
            report["auc"] += [metrics.auc(fpr, tpr)]
            # for agg in f1_calcs:
            #     report[f'f1_{agg}'] += [metrics.f1_score(y_true, y_pred, average=agg, pos_label=pos_label)]
            report["predictions"] += [
                (y_true, y_pred)
            ]  # TODO: make sure this is compatible to code for drawing roc curve

        except Exception as e:
            print(e)
            continue

    print(
        "Mean AUC: %s, std: %s"
        % (np.mean(np.array(report["auc"])), np.std(np.array(report["auc"])))
    )
    return report


def run_cross_eval(
    labelfpath, outfile, verboseoutfile, params
):  # get labels into correct format
    labels = iputils.read_label_csv(labelfpath, threshold=params["threshold"])

    legal_params = utils.remove_illegal_kwargs(params, cross_eval_graphemb)

    report = cross_eval_graphemb(labels, **legal_params)

    with utils.safe_open(outfile, mode="wb") as f:
        pkl.dump(dict(report), f)
        print("Finished saving to", outfile)

    specs = copy.deepcopy(params)
    specs.update(
        {
            "auc": np.mean(np.array(report["auc"])),
            "std_auc": np.std(np.array(report["auc"])),
            "train_no": report["train_no"],
            "test_no": report["test_no"],
            "test_negative_no": report["test_negative_no"],
            "test_positive_no": report["test_positive_no"],
            "labelfpath": labelfpath,
        }
    )

    with utils.safe_open(verboseoutfile, mode="w") as f:
        json.dump(specs, f)
        print("Finished saving to", verboseoutfile)


def main(args):
    # TODO add arg for entry type
    parser = argparse.ArgumentParser()

    parser.add_argument("--labelfile", type=str, help="Path df of user labels")

    parser.add_argument(
        "-o",
        "--outfile",
        type=str,
        help="Path to save results (raw predictions and AUC)",
    )
    parser.add_argument(
        "-v",
        "--verboseoutfile",
        type=str,
        required=False,
        help="Path to save results (AUC and configs)",
    )

    parser.add_argument(
        "--config",
        action="store",
        dest="config",
        type=str,
        required=False,
        help="path to config file",
    )

    args = parser.parse_args(args)
    params = json.load(open(args.config, "r"))

    try:
        run_cross_eval(args.labelfile, args.outfile, args.verboseoutfile, params)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main(sys.argv[1:])
# DEBUG LOCAL
# labelfile = 'data/covid/labels/user_labels.csv'

# rt_edgelist ='data/sample/rt/sample_rt_edgelist.txt'
# trust_edgelist ='data/sample/rt/sample_trust_edgelist.txt'
# labels = iputils.read_label_csv(labelfile)
# report = cross_eval_graphemb(labels, rtedgelist=rt_edgelist,trustedgelist=trust_edgelist, score_type='both',complement=False, kfold=5)

# DEBUG LOCAL WITH CONFIG FILE
# labelfile = 'data/covid/labels/user_labels.csv'
# outfile = 'test/intrinsic_knn.pkl'
# verboseoutfile = 'test/intrinsic_knn.json'
# config = 'config/userdim.json'
# params = json.load(open(config,'r'))
# run_cross_eval(labelfile, outfile, verboseoutfile, params)

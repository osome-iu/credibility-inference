""" Reduce dimension and plot embedding of nodes to see class separation"""

import infopolluter.ip_utils as iputils
import infopolluter.utils as utils
import infopolluter.GraphEmb as graphemb
import matplotlib.pyplot as plt

import os
import pandas as pd
import numpy as np
import sys
import argparse
import json
import pickle as pkl

from sklearn.manifold import TSNE


def tsne_reduc(df):
    #     tsne = TSNE(random_state=1, n_iter=15000, metric="cosine")
    tsne = TSNE(n_components=2, learning_rate="auto", init="random", perplexity=3)
    tsrec = tsne.fit_transform(np.asarray(df["embedding"].tolist()))

    df["x"] = tsrec[:, 0]
    df["y"] = tsrec[:, 1]
    return df


if __name__ == "__main__":
    ABS_PATH = "/N/slate/baotruon/infopolluters"
    DATA_PATH = os.path.join(ABS_PATH, "data")
    PLOT_DIR = "/N/u/baotruon/Carbonate/infopolluters/exps"
    SAMPLEDIR = "/N/slate/baotruon/infopolluters/exp/user_rt/intermediate_sample_node"
    # w2vfile =os.path.join(SAMPLEDIR, "sample_1024_node2vec3.pkl")
    w2vfile = os.path.join(SAMPLEDIR, "sample_32768_node2vec3.pkl")
    # Big file (too big!)
    # w2vfile =os.path.join(DATA_PATH, "cc_size322208_node2vec3.pkl")
    labelfile = os.path.join(DATA_PATH, "user_labels.csv")

    labels = iputils.read_label_csv(labelfile)

    ranker = graphemb.GraphEmb()
    w2v_model = pkl.load(open(w2vfile, "rb"))
    embeddings = graphemb._word2vec_object_to_dict(w2v_model)

    # get labeled embeddings
    ranker.set_embeddings(embeddings)
    # get emb df
    emb_list = []
    for user in embeddings.keys():
        emb_list.append([user, embeddings[user]])

    df = pd.DataFrame(emb_list, columns=["uid", "embedding"])
    # merge with labels
    labeled = pd.merge(df, labels, on="uid", how="left")

    tsrec = tsne_reduc(labeled)

    try:
        tsrec.to_parquet(os.path.join(DATA_PATH, "tsne_sample_32768_node2vec3.parquet"))
    except Exception as e:
        print(e)
        pass

    # # If already saved file:
    # tsrec = pd.read_parquet(os.path.join(DATA_PATH, 'tsne_size322208_node2vec3.parquet'))

    fig, ax = plt.subplots()
    labeled_tsrec = tsrec[~(tsrec["label"].isna())]
    labels = {0: "high-cred", 1: "low-cred"}
    colors = {0: "orange", 1: "purple"}
    for g in labels.keys():
        relative_idx = np.where(labeled_tsrec["label"] == g)
        # npwhere resets index, idx is now relative: range(len(labeled_tsrec)), whereas index of labeled_tsrec is not the full range
        idx = sorted(np.array(labeled_tsrec.index)[relative_idx])
        ax.scatter(
            labeled_tsrec.loc[idx, "x"],
            labeled_tsrec.loc[idx, "y"],
            label=labels[g],
            color=colors[g],
            alpha=0.5,
        )
    ax.legend()
    fig.tight_layout()
    if utils.make_sure_dir_exists(PLOT_DIR, "") is True:
        fpath = os.path.join(PLOT_DIR, "tsne_sample_32768_node2vec3_labeled.pdf")
        plt.savefig(fpath)

    plt.clf()

    fig, ax = plt.subplots()
    tsrec["label"] = tsrec["label"].apply(lambda x: 3 if np.isnan(x) else x)
    labels = {0: "high-cred", 1: "low-cred", 3: "unlabeled"}
    colors = {0: "orange", 1: "purple", 3: "grey"}
    for g in labels.keys():
        idx = np.where(tsrec["label"] == g)
        ax.scatter(
            tsrec.loc[idx, "x"],
            tsrec.loc[idx, "y"],
            label=labels[g],
            color=colors[g],
            alpha=0.4,
        )
    ax.legend()
    fig.tight_layout()
    if utils.make_sure_dir_exists(PLOT_DIR, "") is True:
        fpath = os.path.join(PLOT_DIR, "tsne_sample_32768_node2vec3_all.pdf")
        plt.savefig(fpath)

    print(f"Finish saving figs to {PLOT_DIR}!")

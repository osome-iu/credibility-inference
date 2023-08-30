"""" 
Project the user-domain network to create a cosharing network
    - Input: .parquet file of Weighted Bipartite edgelist (user - domain - number of shares)
    - Output: .parquet file of the Weighted Cosharing edgelist (user-user)
"""
#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
import sparse_dot_topn.sparse_dot_topn as ct
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# TODO: Add logging. Might not be necessary?
def dummy_func(doc):
    return doc


class DocConverter(object):
    def __init__(self):
        self._p1_idx_map = None

    def edge2doc(self, edge_df, p1_col, p2_col, w_col):
        self._p1_idx_map = dict()
        idx = 0
        for p1_node, grp in edge_df.groupby(p1_col):
            self._p1_idx_map[p1_node] = idx
            idx += 1
            yield grp[p2_col].repeat(grp[w_col]).values
        raise StopIteration


def dot_top(A, B, ntop, lower_bound=0):
    """
    from https://towardsdatascience.com/fuzzy-matching-at-scale-84f2bfd0c536
    """
    # force A and B as a CSR matrix.
    # If they have already been CSR, there is no overhead
    A = A.tocsr()
    B = B.tocsr()
    M, _ = A.shape
    _, N = B.shape

    idx_dtype = np.int32

    nnz_max = M * ntop

    indptr = np.zeros(M + 1, dtype=idx_dtype)
    indices = np.zeros(
        nnz_max, dtype=idx_dtype
    )  # Mapping between sparse row format and df
    data = np.zeros(nnz_max, dtype=A.dtype)
    ct.sparse_dot_topn(
        M,
        N,
        np.asarray(A.indptr, dtype=idx_dtype),
        np.asarray(A.indices, dtype=idx_dtype),
        A.data,
        np.asarray(B.indptr, dtype=idx_dtype),
        np.asarray(B.indices, dtype=idx_dtype),
        B.data,
        ntop,
        lower_bound,
        indptr,
        indices,
        data,
    )
    return csr_matrix((data, indices, indptr), shape=(M, N))


def calculate_interaction(
    edge_df, p1_col, p2_col, w_col, node1_col, node2_col, sim_col, sup_col
):
    """
    calculate interactions between nodes in partition 1

    input:
        edge_df : dataframe that contains (p1, p2, w)
        p1_col : column name that represent p1, i.e: uid
        p2_col : column name that represent p2 , e.g: time
        w_col : column name that represent weight, if None it's unweighted
        node1_col, node2_col: col after projection, both are uid
    return:
        interactions : dict { (p1_node1, p1_node2) : interaction_weight }

    assumption:
        1. non-empty inputs: edge_df
        2. p1_node1 < p1_node2 holds for the tuples in interactions
    """
    user_ids, docs = zip(
        *(
            (p1_node, grp[p2_col].repeat(grp[w_col]).values)
            for p1_node, grp in edge_df.groupby(p1_col)
            if len(grp) > 1
        )
    )

    try:
        tfidf = TfidfVectorizer(
            analyzer="word",
            tokenizer=dummy_func,
            preprocessor=dummy_func,
            use_idf=True,
            token_pattern=None,
            lowercase=False,
            sublinear_tf=True,
            min_df=5,
        )

        docs_vec = tfidf.fit_transform(docs)
    except ValueError:
        tfidf = TfidfVectorizer(
            analyzer="word",
            tokenizer=dummy_func,
            preprocessor=dummy_func,
            use_idf=True,
            token_pattern=None,
            lowercase=False,
            sublinear_tf=True,
            min_df=5,
        )
        docs_vec = tfidf.fit_transform(docs)

    results = dot_top(
        docs_vec, docs_vec.T, ntop=docs_vec.toarray().shape[0], lower_bound=0.3
    )
    # non_zero_features = (docs_vec > 0)*1
    # supports = non_zero_features * tfidf.idf_

    # number of tweets/retweets as support
    supports = edge_df.groupby(p1_col)[w_col].sum().to_dict()

    interactions = pd.DataFrame.from_records(
        [
            (
                u1,
                user_ids[u2_idx],
                sim,
                min(supports[user_ids[u1_idx]], supports[user_ids[u2_idx]]),
            )
            for u1_idx, u1 in enumerate(user_ids)
            for u2_idx, sim in zip(
                results.indices[results.indptr[u1_idx] : results.indptr[u1_idx + 1]],
                results.data[results.indptr[u1_idx] : results.indptr[u1_idx + 1]],
            )
            if u1_idx < u2_idx
        ],
        columns=[node1_col, node2_col, sim_col, sup_col],
    )

    return interactions


if __name__ == "__main__":
    ## ARGS
    infile = sys.argv[1]
    outfile = sys.argv[2]

    p1_col = "user_id"
    p2_col = "domain"
    w_col = "weight"
    node1_col = "uid1"
    node2_col = "uid2"
    sim_col = "cosine_sim"
    sup_col = "support"

    # read input
    # edge_df = pd.read_csv(infile, sep=" ", encoding="utf-8")
    edge_df = pd.read_parquet(infile, engine="pyarrow")

    print("Creating one-mode projection of bipartite network..")
    if len(edge_df) > 0 and np.any(edge_df.groupby(p1_col).size() > 1):
        try:
            # do work
            interaction_df = calculate_interaction(
                edge_df, p1_col, p2_col, w_col, node1_col, node2_col, sim_col, sup_col
            )
            # write output
            interaction_df.to_parquet(outfile)
            # also save .txt

            txtout = outfile.replace(".parquet", ".txt")
            interaction_df[["uid1", "uid2", "cosine_sim"]].to_csv(
                txtout, sep=" ", encoding="utf-8", header=False, index=False
            )
        except Exception as e:
            print(e)
    else:
        with open(outfile, "a") as f:
            pass

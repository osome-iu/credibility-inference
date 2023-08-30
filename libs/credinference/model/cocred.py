"""
This module provides the implementation for HITS-like algorithm on bipartite networks.

e.g: CoCred: Propagates the domain labels to rank users. 

Usage:
------
import BipartiteNetwork

# Create an instance of the Bipartite network, set edgelist and labels (optional)
bn = BipartiteNetwork()
bn.set_edgelist(df)

# Generate scores for 2 partitions
dscores, uscores = bn.generate_birank(normalizer="CoCred")
"""

import pandas as pd
import numpy as np
import scipy
import scipy.sparse as spa


def normalize(v):
    norm = np.linalg.norm(v, ord=1)
    if norm == 0:
        return v
    return v / norm


def birank(
    W,
    L,
    normalizer="CoCred",
    alpha=0.85,
    beta=0.85,
    max_iter=1000,
    tol=1.0e-9,
    verbose=False,
):
    """
    Calculate the PageRank of bipartite networks directly.
    See paper https://ieeexplore.ieee.org/abstract/document/7572089/
    for details.
    Different normalizer yields very different results.
    More studies are needed for deciding the right one.

    Input:
        W::scipy's sparse matrix:Adjacency matrix of the bipartite network D*P
        normalizer::string:Choose which normalizer to use, see the paper for details
        alpha, beta::float:Damping factors for the rows and columns
        max_iter::int:Maximum iteration times
        tol::float:Error tolerance to check convergence
        verbose::boolean:If print iteration information

    Output:
         d, p::numpy.ndarray:The BiRank for rows and columns
    """
    print("NORMALIZER: ", normalizer)
    # ensure convergence
    if normalizer == "CoCred" and len(L[L != -1]) == 0:
        raise ValueError(
            "CoCred normalizer requires at least some not None top column labels"
        )

    W = W.astype("float", copy=False)
    WT = W.T

    Kd = scipy.array(W.sum(axis=1)).flatten()
    Kp = scipy.array(W.sum(axis=0)).flatten()
    # avoid divided by zero issue
    Kd[np.where(Kd == 0)] += 1
    Kp[np.where(Kp == 0)] += 1

    Kd_ = spa.diags(1 / Kd)
    Kp_ = spa.diags(1 / Kp)

    if normalizer == "HITS":
        Sp = WT
        Sd = W
    elif normalizer == "CoHITS" or normalizer == "CoCred":
        Sp = WT.dot(Kd_)
        Sd = W.dot(Kp_)
    elif normalizer == "BGRM":
        Sp = Kp_.dot(WT).dot(Kd_)
        Sd = Sp.T
    elif normalizer == "BiRank":
        Kd_bi = spa.diags(1 / scipy.sqrt(Kd))
        Kp_bi = spa.diags(1 / scipy.sqrt(Kp))
        Sp = Kp_bi.dot(WT).dot(Kd_bi)
        Sd = Sp.T

    d0 = np.repeat(1 / Kd_.shape[0], Kd_.shape[0])

    if normalizer == "CoHITS" or normalizer == "CoCred":
        # Initialize domain scores with known labels
        d0[np.where(L == 0)] = 0
        d0[np.where(L == 1)] = 1

    d0 = normalize(d0)
    d_last = d0.copy()
    p0 = np.repeat(1 / Kp_.shape[0], Kp_.shape[0])
    p_last = p0.copy()

    for i in range(max_iter):
        p = alpha * (Sp.dot(d_last)) + (1 - alpha) * p0
        d = beta * (Sd.dot(p_last)) + (1 - beta) * d0

        if normalizer == "CoCred":
            # reset the known domain scores to groundtruth at each iteration
            d[np.where(L == 0)] = 0
            d[np.where(L == 1)] = 1

            d = normalize(d)
            p = normalize(p)
        if normalizer == "HITS":
            # no label
            p = normalize(p)
            d = normalize(d)

        err_p = np.absolute(p - p_last).sum()
        err_d = np.absolute(d - d_last).sum()

        ##TODO: Track score change over time
        if verbose:
            print("--Top score after update: ", d)
            print("Bot score after update: ", p)
            print(
                "Iteration : {}; top error: {}; bottom error: {}".format(
                    i, err_d, err_p
                )
            )
        if err_p < tol and err_d < tol:
            break
        d_last = d
        p_last = p

    return d, p


class BipartiteNetwork:
    """
    Class for handling bipartite networks using scipy's sparse matrix
    Designed to for large networkx, but functionalities are limited
    """

    def __init__(self):
        pass

    def load_edgelist(
        self, edgelist_path, top_col, bottom_col, weight_col="None", sep=","
    ):
        """
        Method to load the edgelist.

        Input:
            edge_list_path::string: the path to the edgelist file
            top_col::string: column of the top nodes
            bottom_col::string: column of the bottom nodes
            weight_col::string: column of the edge weights
            sep::string: the seperators of the edgelist file


        Suppose the bipartite network has D top nodes and
        P bottom nodes.
        The edgelist file should have the format similar to the example:

        top,bottom,weight
        t1,b1,1
        t1,b2,1
        t2,b1,2
        ...
        tD,bP,1

        The edgelist file needs at least two columns for the top nodes and
        bottom nodes. An optional column can carry the edge weight.
        You need to specify the columns in the method parameters.

        The network is represented by a D*P dimensional matrix.
        """

        temp_df = pd.read_csv(edgelist_path, sep=sep)
        self.set_edgelist(temp_df, top_col, bottom_col, weight_col)

    def set_edgelist(self, df, top_col, bottom_col, weight_col=None, label_col=None):
        """
        Method to set the edgelist.

        Input:
            df::pandas.DataFrame: the edgelist with at least two columns
            top_col::string: column of the edgelist dataframe for top nodes
            bottom_col::string: column of the edgelist dataframe for bottom nodes
            weight_col::string: column of the edgelist dataframe for edge weights

        The edgelist should be represented by a dataframe.
        The dataframe eeds at least two columns for the top nodes and
        bottom nodes. An optional column can carry the edge weight.
        You need to specify the columns in the method parameters.
        """
        self.df = df
        self.top_col = top_col
        self.bottom_col = bottom_col
        self.weight_col = weight_col
        self.label_col = label_col
        self._index_nodes()

        ## Create domain label vector to use in network propagation
        if label_col is None:
            self.L = np.zeros(len(self.df["top_index"]))
        else:
            label_df = self.df[[top_col, label_col, "top_index"]].set_index("top_index")
            label_df = label_df[~label_df.index.duplicated(keep="first")]
            label_df.sort_index(ascending=True, inplace=True)
            self.L = label_df[label_col].values

        self._generate_adj()

    def _index_nodes(self):
        # b: map nodes to indices
        """
        Representing the network with adjacency matrix requires indexing the top
        and bottom nodes first
        """
        self.top_ids = pd.DataFrame(
            self.df[self.top_col].unique(), columns=[self.top_col]
        ).reset_index()
        self.top_ids = self.top_ids.rename(columns={"index": "top_index"})

        self.bottom_ids = pd.DataFrame(
            self.df[self.bottom_col].unique(), columns=[self.bottom_col]
        ).reset_index()
        self.bottom_ids = self.bottom_ids.rename(columns={"index": "bottom_index"})

        self.df = self.df.merge(self.top_ids, on=self.top_col)
        self.df = self.df.merge(self.bottom_ids, on=self.bottom_col)

    def _generate_adj(self):
        """
        Generating the adjacency matrix for the birparite network.
        The matrix has dimension: D * P where D is the number of top nodes
        and P is the number of bottom nodes
        """
        if self.weight_col is None:
            # set weight to 1 if weight column is not present
            weight = np.ones(len(self.df))
        else:
            weight = self.df[self.weight_col]
        self.W = spa.coo_matrix(
            (weight, (self.df["top_index"].values, self.df["bottom_index"].values))
        )

    def generate_degree(self):
        """
        This method returns the degree of nodes in the bipartite network
        """
        top_df = self.df.groupby(self.top_col)[self.bottom_col].nunique()
        top_df = top_df.to_frame(name="degree").reset_index()
        bottom_df = self.df.groupby(self.bottom_col)[self.top_col].nunique()
        bottom_df = bottom_df.to_frame(name="degree").reset_index()
        return top_df, bottom_df

    # TODO: Return a dict {node: score}
    def generate_birank(self, **kwargs):
        """
        This method performs BiRank algorithm on the bipartite network and
        returns the ranking values for both the top nodes and bottom nodes.

        Input: adj matrix of a bipartite network
        Output: vectors of ranking for the partitions

        """
        # b: add self.L for labels
        d, p = birank(self.W, self.L, **kwargs)
        top_df = self.top_ids.copy()
        bottom_df = self.bottom_ids.copy()
        top_df[self.top_col + "_birank"] = d
        bottom_df[self.bottom_col + "_birank"] = p
        return (
            top_df[[self.top_col, self.top_col + "_birank"]],
            bottom_df[[self.bottom_col, self.bottom_col + "_birank"]],
        )

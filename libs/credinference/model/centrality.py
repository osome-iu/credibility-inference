"""
Wrapper to evaluate Link-based algorithms (LoCred, PageRank Trust, Trustrank, etc.)
Implements fit, predict() similarly to sklearn template, hence is compatible to sklearn.model_selection.cross_validate()
"""
import numpy as np
import pandas as pd
from .link_prop import locred, pagerank_trust, reputation_scaling, trustrank
from copy import deepcopy
from sklearn.base import BaseEstimator
from credinference.util import user_col


class CentralityRanker(BaseEstimator):
    """
    A class to perform and evaluate centrality measure ranking.
        - method (str): {'locred','ppt', 'pt', 'trustrank', 'reputation_scaling'}
    """

    def __init__(
        self,
        graph=None,
        vertex_df=None,
        user_info="",
        method: str = "locred",
        weight_col="weight",
        alpha=None,
        vary_alpha=None,
    ):
        """
        - user_info: Dataframe with columns: ["user_id", "user_score", "confidence"]
        """
        self.graph = graph
        self.vertex_df = vertex_df
        self.user_info = user_info
        self.method = method
        self.weight_col = weight_col
        self.alpha = alpha if alpha is not None else 0.8
        self.vary_alpha = vary_alpha if vary_alpha is not None else False

        return

    def _set_attributes(self):
        """
        Set other attributes that were not set during init
        """

        self.alpha1_ = self.alpha
        if self.vary_alpha is True:
            self.alpha2_ = 1 - self.alpha1_
        else:
            self.alpha2_ = self.alpha

        self.score_name_ = self.method
        # col with labels to personalized the centrality measure
        self.label_name_ = "label"
        return

    def fit(self, X, y):
        """
        X: np.array (n,2): index of fit users 

        """
        self._set_attributes()

        # Assign labels to nodes in graph based on user dummy labels from label_dict
        # Only use label with 100% confidence:
        graph = deepcopy(self.graph)

        # self.data is a list of vertices and their labels
        # self.data = self.user_info.copy()
        # NOTE: Some users in the graph won't have score information because they are retweeted user (who we don't have domain-sharing info for)
        # use labels of known users
        fit_df = pd.DataFrame(X, columns=["vertex ID", "fit_label"])
        self.data = self.user_info.merge(fit_df, on="vertex ID", how="left")
        self.data = self.data.rename(columns={"fit_label": "dummy_label"})
        print(f"No. accounts used to fit: {len(fit_df)}")
        # print("Data used in fit: ")
        # print(self.data.head(10))

        # add labels to graph
        print(f"No. labels used to fit: {self.data['dummy_label'].notna().sum()}")
        graph.vs[self.label_name_] = self.data["dummy_label"].values

        # Ranking
        if self.score_name_ == "locred":
            centrality = locred
            args = {"alpha": self.alpha}
        elif self.score_name_ == "ppt":
            centrality = pagerank_trust
            args = {"alpha": self.alpha, "personalized": True}
        elif self.score_name_ == "pt":
            centrality = pagerank_trust
            args = {"alpha": self.alpha, "personalized": False}
        elif self.score_name_ == "trustrank":
            centrality = trustrank
            args = {
                "alpha1": self.alpha1_,
                "alpha2": self.alpha2_,
                "num_seeds": int(self.graph.vcount() * 0.3),
            }
        elif self.score_name_ == "reputation_scaling":
            centrality = reputation_scaling
            args = {"alpha1": self.alpha1_, "alpha2": self.alpha2_}

        results = centrality(graph, weight_col=self.weight_col, **args)
        self.data["prediction"] = self.data[user_col].apply(
            lambda user: results[user] if user in results.keys() else np.nan
        )

        return

    def predict(self, X):
        """
        Return preds: a subset of user_info df with an extra column "prediction"

        """
        pred_vertices = X[:, 0].tolist()
        # Get ranking for predict users:
        preds = self.data[self.data["vertex ID"].isin(pred_vertices)]
        assert preds["dummy_label"].isna().sum() == len(X)

        # TODO: for all other methods except for "locred": high score == good
        # but for our evaluation 1 = positive class (infopolluters), so we need to flip the scores? i.e: 1-score
        ## b: locred: 1- positive (Bad), 0- negative; PR: 1- neg (Good), 0- positive (bad)
        # so we need to flip the scores

        if self.method != "locred":
            preds["prediction"] = preds["prediction"].apply(lambda x: 1 - x)

        # preds = preds[["prediction", "true_label"]]
        # print(preds.describe())
        # print(f'No nan predictions: {preds["prediction"].isna().sum()}')
        # print(f'No nan true_label: {preds["true_label"].isna().sum()}')
        preds = preds.rename(columns={"prediction": "y_pred", "true_label": "y_true"})
        return preds

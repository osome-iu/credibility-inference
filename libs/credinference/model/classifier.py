"""
Provide wrapper to evaluate embedding classification (user-split)
Implements fit, predict() similarly to sklearn template, hence is compatible to sklearn.model_selection.cross_validate()
"""
from copy import deepcopy
from sklearn.base import BaseEstimator
from .clf_util import *
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    accuracy_score,
)
import matplotlib.pyplot as plt


def find_optimal_cutoff(target, predicted, pos_label, plot=False, save_fig=False):
    """
    Find the optimal probability cutoff point for a classification model related to event rate
    Verbose with option to plot for debugging
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------
    list type, with optimal cutoff value

    """
    # target = [1 if val is False else 0 for val in target]
    fpr, tpr, thresholds = metrics.roc_curve(target, predicted, pos_label=pos_label)
    roc = pd.DataFrame(
        data={"fpr": fpr, "tpr": tpr, "threshold": thresholds, "j_index": tpr - fpr}
    )

    # optimal = roc.sort_values(by="j_index", ascending=False)

    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    plot_data = roc[roc.threshold <= 1].sort_values(by="threshold")
    if plot is True:
        fig, ax1 = plt.subplots()

        ax2 = ax1.twinx()
        ax1.plot(plot_data["threshold"], plot_data["fpr"], color="blue", marker="^")
        ax1.set_ylim(ax1.get_ylim()[::-1])
        ax1.set_xlabel("Threshold")

        plt.scatter(
            optimal_threshold,
            plot_data[plot_data["threshold"] == optimal_threshold]["tpr"],
            marker="o",
            color="black",
            label="Best",
        )

        ax2.plot(plot_data["threshold"], plot_data["tpr"], color="orange", marker=".")

        ax1.set_ylabel("False Positive Rate", color="blue")
        ax2.set_ylabel("True Positive Rate", color="orange")
        plt.title(f"Optimal binary threshold = {np.round(optimal_threshold, 3)}")
        plt.legend()
        if save_fig is True:
            plt.savefig("optimal_threshold.png", dpi=300)
        else:
            plt.show()

    predicted_binary = [1 if i >= optimal_threshold else 0 for i in predicted]
    return predicted_binary


def optimal_threshold(target, predicted, pos_label, metric="f1"):
    """
    Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations
    pos_label: locred: 1- positive (Bad), 0- negative; PR: 1- neg (Good), 0- positive
    metric: metric to optimize on {auc, f1}, default: f1

    Returns
    -------
    list type, with optimal cutoff value

    """
    neg_label = 0 if pos_label == 1 else 1

    def binarize(x, threshold):
        if x >= threshold:
            return pos_label
        else:
            return neg_label

    optimal_metric = 0
    predicted_binary = []
    if metric == "f1":
        precisions, recalls, thresholds = metrics.precision_recall_curve(
            target, predicted
        )
        for threshold in thresholds:
            binary = [binarize(i, threshold) for i in predicted]
            f1 = metrics.f1_score(target, binary)
            if f1 > optimal_metric:
                optimal_metric = f1
                predicted_binary = binary
    elif metric == "auc":
        fpr, tpr, thresholds = metrics.roc_curve(target, predicted, pos_label=pos_label)
        optimal_idx = np.argmax(tpr - fpr)
        threshold = thresholds[optimal_idx]
        predicted_binary = [binarize(i, threshold) for i in predicted]
        optimal_metric = metrics.roc_auc_score(y_true=target, y_score=predicted_binary)
        # metrics.auc(fpr[optimal_idx], tpr[optimal_idx])
    else:
        raise ValueError("Unknown metric to optimize on. Must be either 'auc' or 'f1'")
    print(
        f"Optimal {metric}={np.round(optimal_metric,4)} (threshold={np.round(threshold,3)})"
    )
    return predicted_binary


def summarize_evaluation(
    target, predicted, pos_label=1, find_optimal=False, metric="f1"
):
    """
    Return evaluation metrics for the prediction.
    Print classification report
    find_optimal: If True, find optimal cutoff threshold for classifier.
    
    binary: whether to use raw prediction scores (continuous) for AUC calculation. if True, returns binary class predictions
    """
    if find_optimal == True:
        predicted = optimal_threshold(
            target, predicted, pos_label=pos_label, metric=metric
        )

    cm = confusion_matrix(target, predicted)
    # if len(set(predicted)) < 2:
    #     print("Only 1 class was predicted, skipping..")
    #     return None
    precision, recall, f1_macro, support = metrics.precision_recall_fscore_support(
        y_true=target, y_pred=predicted, average="macro"
    )
    report = {
        "accuracy": metrics.accuracy_score(y_true=target, y_pred=predicted),
        "auc": metrics.roc_auc_score(y_true=target, y_score=predicted),
        "f1": metrics.f1_score(y_true=target, y_pred=predicted),
        "precision": precision,
        "recall": recall,
        "f1_macro": f1_macro,
        "tn": cm[0, 0],
        "fp": cm[0, 1],
        "fn": cm[1, 0],
        "tp": cm[1, 1],
    }

    clf_report = classification_report(y_true=target, y_pred=predicted)
    print(clf_report)
    return report


class PopulationClassifier(BaseEstimator):
    """
    Class for performing classification with Graph and Text embedding 
    """

    def __init__(
        self,
        population=None,
        predictor="knn",
        clf=None,
        pred_feats=None,
        labeller=None,
        clf_attribute_name="label",
        clf_prob_attribute_name="probability",
        positive_label=1,
    ):
        """
        Initializes a new classifier.

        Args:
            population (list, optional): List of users to be included in the training population. Defaults to None.
            user_share_path (str, optional): Path to user share file. Defaults to "".
            domain_idx_path (str, optional): Path to domain index file. Defaults to "".
            clf (object, optional): Pre-trained scikit-learn classifier. Defaults to None.
            pred_feats (list, optional): List of predictor features. Defaults to None.
            labeller (function, optional): Function to label users. Defaults to None.
            clf_attribute_name (str, optional): Name of the classifier attribute for labels. Defaults to "label".
            clf_prob_attribute_name (str, optional): Name of the classifier attribute for probabilities. Defaults to "probability".
            positive_label (int, optional): Positive label for classifier. Defaults to 1.
            NOTE: domain_idx df has to have column "Score"
        Returns:
            None
        """
        print("**INIT Population Classifier**")
        self.population = population
        # self.label_path = label_path
        self.pred_feats = pred_feats
        self.positive_label = positive_label
        if labeller is None:
            labeller = lambda user: user.meta["true_label"] == positive_label
        self.labeller = labeller

        if predictor == "dectree":
            predictor = DecisionTreeClassifier()
        elif predictor == "randforest":
            predictor = RandomForestClassifier()
        elif predictor == "logreg":
            predictor = LogisticRegression(solver="liblinear")
        elif (predictor == "knn") or (predictor is None):
            predictor = KNeighborsClassifier(n_neighbors=10)

        self.predictor = predictor

        if clf is None:
            clf = Pipeline(
                [
                    ("standardScaler", StandardScaler(with_mean=False)),
                    ("clf", self.predictor),
                ]
            )
        print(
            f"Initialized default classification model. Steps: {list(clf.named_steps.values())}"
        )
        self.clf = clf
        self.clf_attribute_name = clf_attribute_name
        self.clf_prob_attribute_name = clf_prob_attribute_name
        self.positive_label = positive_label

        return

    def fit(self, X_user, y_user=None):
        """
        X: np.array (n,2)
            first col: user indices to label 
            second col: confidence
        y: np.array (n,) labels
        Get the population (X) for which X_domain information is available with 100 confidence

        Fit the internal classifier model on the vector matrix (X,y) that represents these users 
        the Corpus components, with an optional selector that selects for objects to be fit on.
        """
        # b: feeding clf with domains label to result in feature matrix X
        # so when pred is called (knowing more labels), known_domain_labels should be updated, not initialized from nan again

        fit_selector = lambda user: user.meta["user_index"] in X_user

        fit_users = [user for user in self.population.iter_users(selector=fit_selector)]
        print(f"Fitting.. No. accounts: {len(fit_users)}")

        X, y, _ = extract_vector_feats_and_label(
            self.population,
            pred_feats=self.pred_feats,
            labeller=self.labeller,
            selector=fit_selector,
        )
        self.clf.fit(X, y)

        return

    def predict(self, X_user):
        """
        X: np.array (n,2)
            first col: domain indices to label 
            second col: scores

        Get the population (X) for which X_domain information is available with 100 confidence
        Get the feature for these users 
        Return the probability score for test users (those whose label was hidden)
        """
        population = self.population

        predict_selector = lambda user: user.meta["user_index"] in X_user
        predict_users = [
            user for user in population.iter_users(selector=predict_selector)
        ]
        print(f"Predicting.. Number of users in prediction {len(predict_users)}")

        X, _, uids = extract_vector_feats_and_label(
            population,
            pred_feats=self.pred_feats,
            labeller=self.labeller,
            selector=predict_selector,
        )
        objs = [obj for obj in population.iter_users(selector=predict_selector)]

        try:
            clfs, clfs_probs = self.clf.predict(X), self.clf.predict_proba(X)[:, 1]
        except:
            clfs, clfs_probs = self.clf.predict(X), []
        # TODO: check that idx is indeed user correct id
        for idx, (clf, clf_prob) in enumerate(list(zip(clfs, clfs_probs))):
            obj = objs[idx]
            obj.add_meta(self.clf_attribute_name, clf)
            obj.add_meta(self.clf_prob_attribute_name, clf_prob)

        return population

from .population import Population
from .user import User
from typing import List, Callable
import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np
from credinference.util import warn, get_dict_val
from collections import defaultdict
from .matrix import ConvoKitMatrix
from sklearn.impute import SimpleImputer, KNNImputer


def extract_feats_from_obj(obj: User, pred_feats: List[str]):
    """
    Assuming feature data has at most one level of nesting, i.e. meta['height'] = 1, and meta['grades'] = {'prelim1': 99,
    'prelim2': 75, 'final': 100}
    Extract the features values from a Population object
    :param obj: Population object
    :param pred_feats: list of features to extract metadata for
    :return: dictionary of predictive feature names to values
    """
    # TODO: simplify using method below
    retval = dict()
    for feat_name in pred_feats:
        feat_val = obj.meta[feat_name]
        if type(feat_val) == dict:
            retval.update(feat_val)
        else:
            retval[feat_name] = feat_val
    return retval


def extract_feat_from_obj(obj: User, feat_name: str):
    """
    Assuming feature data has at most one level of nesting, i.e. meta['height'] = 1, and meta['grades'] = {'prelim1': 99,
    'prelim2': 75, 'final': 100}
    Extract the features values from a Population object
    :param obj: Population object
    :param feat_name: name of feature to extract metadata for
    :return: dictionary of predictive feature names to values
    """
    feat_val = get_dict_val(obj.meta, key_list=[feat_name])
    return feat_val


def extract_feats_dict(
    corpus: Population,
    obj_type: str,
    pred_feats: List[str],
    selector: Callable[[User], bool] = lambda x: True,
):
    """
    Extract features dictionary from a corpus
    :param corpus: target corpus
    :param obj_type: Population object type
    :param pred_feats: list of features to extract metadata for
    :param selector: function to select for Population objects to extract features from
    :return: dictionary mapping object id to a dictionary of predictive features
    """
    users = [u for u in corpus.iter_users(selector)]
    # Bao: example: obj_id_to_feats = {'uid1': {"node2vec":[], "locred": 1}, 'uid2': {"node2vec":[], "locred": 0}}
    obj_id_to_feats = {
        obj.id: extract_feats_from_obj(obj, pred_feats)
        for obj in corpus.iter_users(selector)
    }

    return obj_id_to_feats


def extract_feats(
    corpus: Population,
    obj_type: str,
    pred_feats: List[str],
    selector: Callable[[User], bool] = lambda x: True,
):
    """
    Extract a matrix representation of Population objects' features from corpus
    :param corpus: target corpus
    :param obj_type: Population object type
    :param pred_feats: list of features to extract metadata for
    :param selector: function to select for Population objects to extract features from
    :return: matrix of Population objects' features
    """
    obj_id_to_feats = extract_feats_dict(corpus, obj_type, pred_feats, selector)
    feats_df = pd.DataFrame.from_dict(obj_id_to_feats, orient="index")
    return csr_matrix(feats_df.values)


def extract_label_dict(
    corpus: Population,
    obj_type: str,
    labeller: Callable[[User], bool],
    selector: Callable[[User], bool] = lambda x: True,
):
    """
    Generate dictionary mapping Population object id to label from corpus
    :param corpus: target corpus
    :param obj_type: Population object type
    :param labeller: function that takes a Population object as input and outputs its label
    :param selector: function to select for Population objects to extract features from
    :return: dictionary mapping Population object id to label
    """
    obj_id_to_label = dict()
    for obj in corpus.iter_users(selector):
        obj_id_to_label[obj.id] = {"y": 1} if labeller(obj) else {"y": 0}

    return obj_id_to_label


def extract_feats_and_label(
    corpus: Population,
    obj_type: str,
    pred_feats: List[str],
    labeller: Callable[[User], bool],
    selector: Callable[[User], bool] = None,
):
    """
    Extract matrix of predictive features and numpy array of labels from corpus
    :param corpus: target Population
    :param obj_type: Population object type
    :param pred_feats: list of features to extract metadata for
    :param labeller: function that takes a Population object as input and outputs its label
    :param selector: function to select for Population objects to extract features from
    :return: matrix of predictive features and numpy array of labels
    """
    obj_id_to_feats = extract_feats_dict(corpus, obj_type, pred_feats, selector)
    obj_id_to_label = extract_label_dict(corpus, obj_type, labeller, selector)
    # b: X_df: df with at least 2 columns: 'uid' and 'feature'
    # y_df: df with at least 2 columns: 'uid' and 'label'
    X_df = pd.DataFrame.from_dict(obj_id_to_feats, orient="index")
    y_df = pd.DataFrame.from_dict(obj_id_to_label, orient="index")

    X_y_df = pd.concat([X_df, y_df], axis=1, sort=False)

    print(X_df.head())
    print(y_df.head())
    print(X_y_df.head())
    y = X_y_df["y"]
    X = X_y_df.drop(columns="y")
    X = X.astype("float64")

    return csr_matrix(X.values), np.array(y)


def extract_vector_feats_and_label(corpus, pred_feats, labeller, selector):
    # b: general case for extract feats and extract vectors
    # b: Only works for binary case
    """
    Extract matrix of predictive features and numpy array of labels from corpus
    :param corpus: target Population
    :param obj_type: Population object type
    :param pred_feats: list of features to extract metadata for
    :param labeller: function that takes a User object as input and outputs its label
    :param selector: function to select for Population objects to extract features from
    :return: matrix of predictive features and numpy array of labels
    """
    obj_ids = []
    y_bool = []
    # feat_dict contains feature name, value of that feature matching obj_ids
    # {"node2vec":[], "locred": 1}
    feat_dict = defaultdict(lambda: [])
    for obj in corpus.iter_users(selector):
        obj_ids.append(obj.id)
        y_bool.append(labeller(obj))
        for feat in pred_feats:
            feature_vec = extract_feat_from_obj(obj, feat)
            if type(feature_vec) != np.ndarray:
                feature_vec = np.array([feature_vec])
            feat_dict[feat].append(feature_vec)
    matrices = []
    # get feature values for each user
    for feat, values in feat_dict.items():
        feat_shape = [val for val in values if val is not None][0].shape
        nans = np.empty(feat_shape)
        nans.fill(np.nan)
        vals = [nans if x is None else x for x in values]
        X = np.array(vals)
        # b: use Simple imputer or KNN imputer
        imputer = SimpleImputer()
        # imputer = KNNImputer(n_neighbors=2, weights="uniform")
        matrix = imputer.fit_transform(X)
        columns = [f"{feat}_{idx}" for idx in range(matrix.shape[1])]
        matrix = ConvoKitMatrix(name=feat, matrix=matrix, ids=obj_ids, columns=columns)
        matrices.append(matrix)
    # combine multiple features together
    if len(pred_feats) > 1:
        all_features = ConvoKitMatrix.hstack(
            name="_".join(pred_feats), matrices=matrices
        )
    else:
        all_features = matrix

    X = all_features.matrix
    y = np.array([1 if i == True else 0 for i in y_bool])
    return X, y, obj_ids


# def extract_vector_feats_and_label(
#     corpus, obj_type, vector_name, columns, labeller, selector
# ):
#     # if ((corpus is None) and (objs is None)) or ((corpus is not None) and (objs is not None)):
#     #     raise ValueError("This function takes in either a Population or a list of speakers / utterances / conversations")
#     #
#     # if corpus:
#     #     print("Using corpus objects...")
#     #     objs = list(corpus.iter_users(selector))
#     # else:
#     #     assert objs is not None
#     #     print("Using input list of corpus objects...")
#     objs = list(corpus.iter_users(selector))
#     obj_ids = [obj.id for obj in objs]
#     y = np.array([labeller(obj) for obj in objs])
#     X = corpus.get_vector_matrix(vector_name).get_vectors(obj_ids, columns)

#     return X, y


def get_coefs_helper(clf, feature_names: List[str] = None, coef_func=None):
    """
    Get dataframe of classifier coefficients. By default, assumes it is a pipeline with a logistic regression component
    :param clf: classifier model
    :param feature_names: list of feature names to get coefficients for
    :param coef_func: function for accessing the list of coefficients from the classifier model
    :return: DataFrame of features and coefficients, indexed by feature names
    """
    if coef_func is None:
        try:
            coefs = clf.named_steps["logreg"].coef_[0].tolist()
        except AttributeError:
            warn(
                "Classifier is not a pipeline with a logistic regression component, so default coefficient getter function"
                " did not work. Choose a valid coef_func argument."
            )
            return
    else:
        coefs = coef_func(clf)

    assert len(feature_names) == len(coefs)
    feats_coefs = sorted(
        list(zip(feature_names, coefs)), key=lambda x: x[1], reverse=True
    )
    return (
        pd.DataFrame(feats_coefs, columns=["feat_name", "coef"])
        .set_index("feat_name")
        .sort_values("coef", ascending=False)
    )

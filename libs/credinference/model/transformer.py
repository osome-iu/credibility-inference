"""
Does 1 thing: modify/add a metadata field to a Population 
(nodeembedding_vec, PR_score, etc.)
"""

from abc import ABC, abstractmethod
from .population import Population


class Transformer(ABC):
    """
    Abstract base class for modules that take in a Population and modify the Population
    and/or extend it with additional information, imitating the scikit-learn
    Transformer API. Exposes ``fit()`` and ``transform()`` methods. ``fit()`` performs any
    necessary precomputation (or “training” in machine learning parlance) while
    ``transform()`` does the work of actually computing the modification and
    applying it to the corpus.

    All subclasses must implement ``transform()``;
    subclasses that require precomputation should also override ``fit()``, which by
    default does nothing. Additionally, the interface also exposes a
    ``fit_transform()`` method that does both steps on the same Population in one line.
    By default this is implemented to simply call ``fit()`` followed by ``transform()``,
    but designers of Transformer subclasses may also choose to overwrite the
    default implementation in cases where the combined operation can be
    implemented more efficiently than doing the steps separately.
    """

    def fit(self, corpus: Population, y=None, **kwargs):
        """Use the provided Population to perform any precomputations necessary to
        later perform the actual transformation step.

        :param corpus: the Population to use for fitting

        :return: the fitted Transformer
        """
        return self

    @abstractmethod
    def transform(self, corpus: Population, **kwargs) -> Population:
        """Modify the provided corpus. This is an abstract method that must be
        implemented by any Transformer subclass

        :param corpus: the Population to transform

        :return: modified version of the input Population. Note that unlike the
            scikit-learn equivalent, ``transform()`` operates inplace on the Population
            (though for convenience and compatibility with scikit-learn, it also
            returns the modified Population).
        """
        pass

    def fit_transform(self, corpus: Population, y=None, **kwargs) -> Population:
        """Fit and run the Transformer on a single Population.

        :param corpus: the Population to use

        :return: same as transform
        """
        self.fit(corpus, y=y, **kwargs)
        return self.transform(corpus, **kwargs)

    def summarize(self, corpus: Population, **kwargs):
        pass

    # """
    # Base class for Infopolluters models. Defines the common functions that the children classes
    # should have.
    # Classes for specific model can inheret this base class.
    # """

    # def __init__(self, input_fpath):
    #     """
    #     This function initializes the instance by reading the input file path

    #     Parameters:
    #         - input_fpath (str): the graph or text embedding
    #     """

    #     if input_fpath is None:
    #         raise ValueError("The post object cannot be None")

    #     self.input_file = input_fpath

    def set_embeddings(self, embeddings):
        """
        Set `embeddings` for this model.
        embeddings (dict): {uid: embedding_vec}
        The index for the embeddings is fixed and is used as a reference for add labels and compare ranking
        """
        raise NotImplementedError

    def add_labels(self, labels, undersample=True):
        """
        Add labels for all nodes.
        keep the order of the nodes of the graph fixed (and matches self.embeddings.keys())
        """
        raise NotImplementedError

    def get_known_labels(self):
        """
        Return known labels
        """
        raise NotImplementedError

    def get_ranking(self):
        """
        Return the ranking of users in this model
        """
        raise NotImplementedError

    def __repr__(self):
        """
        Define the representation of the model.
        """
        return f"<{self.__class__.__name__}() model>"

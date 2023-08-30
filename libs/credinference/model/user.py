from typing import Dict, List, Optional
from credinference.util import *
from credinference.util import get_dict_val


class User:
    """Represents a single user in the dataset.

    :param id: the unique id of the user.
    :param meta: A dictionary-like view object providing read-write access to
        utterance-level metadata.
    """

    def __init__(
        self,
        id: Optional[str] = None,
        vectors: List[str] = None,
        tweets: Optional[list] = None,  # tweets are list of dicts
        meta: Optional[Dict] = None,
    ):
        self.obj_type = "user"
        self.id = id
        self.vectors = vectors if vectors is not None else []
        self.tweets = tweets if tweets is not None else []
        if meta is None:
            meta = dict()
        self._meta = self.init_meta(meta)
        return

    def to_dict(self):
        return {
            "id": self.id,
            "vectors": self.vectors,
            "meta": self.meta if type(self.meta) == dict else self.meta.to_dict(),
        }

    def get_id(self):
        return self._id

    def set_id(self, value):
        if not isinstance(value, str) and value is not None:
            self._id = str(value)
            warn(
                "{} id must be a string. ID input has been casted to a string.".format(
                    self.obj_type
                )
            )
        else:
            self._id = value

    id = property(get_id, set_id)

    # def __eq__(self, other):
    #     if type(self) != type(other): return False
    #     # do not compare 'utterances' and 'conversations' in Speaker.__dict__. recursion loop will occur.
    #     self_keys = set(self.__dict__).difference(['_owner', 'meta', 'utterances', 'conversations'])
    #     other_keys = set(other.__dict__).difference(['_owner', 'meta', 'utterances', 'conversations'])
    #     return self_keys == other_keys and all([self.__dict__[k] == other.__dict__[k] for k in self_keys])
    def init_meta(self, meta):
        metadata = {}
        for key, value in meta.items():
            metadata[key] = value
        return metadata
        # if isinstance(meta, Meta):
        #     return meta

        # meta_data = Meta()
        # for key, value in meta.items():
        #     meta_data[key] = value
        # return meta_data

    def get_meta(self, meta: str):
        """
        Extract the metadata from user. Return nested metadata.
        """
        return get_dict_val(self.meta, key_list=[meta])

    def get_all_meta(self):
        return self._meta

    def set_meta(self, new_meta):
        self._meta = self.init_meta(new_meta)

    meta = property(get_all_meta, set_meta)

    def retrieve_meta(self, key: str):
        """
        Retrieves a value stored under the key of the metadata of corpus object
        :param key: name of metadata attribute
        :return: value
        """
        return self.meta.get(key, None)

    def add_meta(self, key: str, value) -> None:
        """
        Adds a key-value pair to the metadata of the corpus object
        :param key: name of metadata attribute
        :param value: value of metadata attribute
        :return: None
        """
        new_meta = self.meta
        new_meta.update({key: value})
        # new_meta.update(Meta(meta={key: value}))
        self.set_meta(new_meta)

    def get_vector(
        self,
        vector_name: str,
        as_dataframe: bool = False,
        columns: Optional[List[str]] = None,
    ):
        """
        Get the vector stored as `vector_name` for this object.
        :param vector_name: name of vector
        :param as_dataframe: whether to return the vector as a dataframe (True) or in its raw array form (False). False
            by default.
        :param columns: optional list of named columns of the vector to include. All columns returned otherwise. This
            parameter is only used if as_dataframe is set to True
        :return: a numpy / scipy array
        """
        if vector_name not in self.vectors:
            raise ValueError(
                "This {} has no vector stored as '{}'.".format(
                    self.obj_type, vector_name
                )
            )

        return self.owner.get_vector_matrix(vector_name).get_vectors(
            ids=[self.id], as_dataframe=as_dataframe, columns=columns
        )

    def add_vector(self, vector_name: str):
        """
        Logs in the Corpus component object's internal vectors list that the component object has a vector row
        associated with it in the vector matrix named `vector_name`.
        Transformers that add vectors to the Corpus should use this to update the relevant component objects during
        the transform() step.
        :param vector_name: name of vector matrix
        :return: None
        """
        if vector_name not in self.vectors:
            self.vectors.append(vector_name)

    def has_vector(self, vector_name: str):
        return vector_name in self.vectors

    def delete_vector(self, vector_name: str):
        """
        Delete a vector associated with this Corpus component object.
        :param vector_name:
        :return: None
        """
        self.vectors.remove(vector_name)

    def __str__(self):
        return "{}(id: {}, vectors: {}, meta: {})".format(
            self.obj_type.capitalize(), self.id, self.vectors, self.meta
        )

    def __hash__(self):
        return hash(self.obj_type + str(self.id))

    def __eq__(self, other):
        if not isinstance(other, User):
            return False
        try:
            return (
                self.id == other.id
                and self.vectors == other.vectors
                and self.tweets == other.tweets
                # TODO: add compare meta
            )
        except AttributeError:
            return self.__dict__ == other.__dict__

    # def __str__(self):
    #     return "User(id: {}, vectors: {}, meta: {})".format(
    #         repr(self.id),
    #         self.vectors,
    #         self.meta,
    #     )

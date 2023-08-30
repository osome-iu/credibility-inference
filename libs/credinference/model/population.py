# from tqdm import tqdm
from typing import (
    List,
    Dict,
    Callable,
    Set,
    Generator,
    Optional,
)
import random
import credinference.util as util
import os
from .user import User
import pandas as pd
import igraph


# TODO: Users need to have uniform metadata fields
class Population:
    """
    Represents a dataset, which is constructed from a list of users.
    If initialized with an edgelist, the population has a graph attribute which is an igraph.Graph instance

    :param filename: Path to a folder containing a Population or to an users.jsonl / users.json file to load
    :param users: list of users to initialize Population from
    :param user_list: list of user names to initialize Population from (NOTE: this is not used anywhere right now. Might be deleted)
    :param preload_vectors: list of names of vectors to be preloaded from directory; by default,
        no vectors are loaded but can be loaded any time after Population initialization (i.e. vectors are lazy-loaded)
    :ivar meta_index: index of Population metadata
    :ivar vectors: the vectors stored in the Population
    :ivar Population_dirpath: path to the directory the Population was loaded from
    """

    def __init__(
        self,
        filename: Optional[str] = None,
        edgefilename: Optional[str] = None,
        users: Optional[List[User]] = None,
        user_list: Optional[List[User]] = None,
    ):
        self.meta = []
        self.users = dict()

        # Initialize with edgelist
        if edgefilename is not None:
            graph, vertex_df = self.graph_from_file(edgefilename)
            for _, row in vertex_df.iterrows():
                user = User(
                    id=row["name"],
                    meta={"uid": row["name"], "graph_id": str(row["vertex ID"])},
                )
                self.users[user.id] = user
            self.graph = graph
            self.add_meta_field("uid")
            self.add_meta_field("graph_id")

        # Initialize with list of User objs
        elif users is not None:
            self.users = {u.id: u for u in users}
        elif user_list is not None:
            self.users = {name: User(id=name) for name in user_list}

        return

    def graph_from_file(self, edgelist):
        """
        Make igraph instance of network
        type = ['graphml', 'graphtxt']
        """

        _, extension = os.path.splitext(edgelist)

        if extension == ".graphml":
            graph = igraph.Graph.Read_GraphML(edgelist)
        elif extension == ".gml":
            graph = igraph.Graph.Graph.Read_GML(edgelist)
        elif extension == ".txt":
            graph = igraph.Graph.Read_Ncol(edgelist, weights=True, directed=True)

        elif extension == ".parquet":
            df = pd.read_parquet(edgelist, engine="pyarrow")
            graph = igraph.Graph.TupleList(
                df.itertuples(index=False),
                directed=True,
                weights=True,  # only True if column "weight" exists
                # edge_attrs="weight", #alternative way to specify weight col
            )

        # return df with 2 cols: [vertex ID, name]
        vertex_df = graph.get_vertex_dataframe().reset_index()

        return graph, vertex_df

    def add_labels(self, labels: Dict):
        """
        Add metadata `label` for users with known label
        labels (dict): mapping between account id and label
        """
        # add label to users
        self.add_user_metadata(attr_key="label", attr_name="label", attr_vals=labels)
        return

    # def add_labels(self, labels: Dict):
    #     """
    #     Add metadata `label` for users with known label
    #     labels (dict): mapping between account id and label
    #     """
    #     # add label to users
    #     for user in self.iter_users():
    #         label = labels[user.id]["label"] if user.id in labels.keys() else None
    #         user.add_meta("label", label)
    #         self.users[user.id] = user
    #     self.add_meta_field("label")

    #     return

    # TODO: Add user metadata using a dictionary.
    def add_user_metadata(
        self, attr_name: str, attr_vals: Dict, attr_key=None, default=None
    ):
        """
        Use to add graph emb feature, or locred score for a user.
        attr_vals (dict): mapping between account id and metadata value
        attr_name: attribute name
        attr_key: the value of the attribute may be further nested in a dict.
        if value not provided for a user, use default value
        """
        for user in self.iter_users():
            if attr_key is None:
                value = util.get_dict_val(
                    attr_vals, key_list=[user.id], default=default
                )
            else:
                value = util.get_dict_val(
                    attr_vals, key_list=[user.id, attr_key], default=default
                )
            # value = (
            #     attr_vals[user.id][attr_name] if user.id in attr_vals.keys() else None
            # )
            user.add_meta(attr_name, value)
            self.users[user.id] = user
        self.add_meta_field(attr_name)

        return

    def hide(self, train_size=0.8, selector=lambda x: x.meta["label"] != None):
        # TODO:return a dummy dict mapp
        # Make test set: Add metadata field 'hidden' to a subset of users, depending on train_size or selector
        # add 'dummy_label' to all hidden users (to use in labeling the graph)
        if "label" not in self.meta:
            raise KeyError(
                "Hidding label for training failed because Population has not been labeled."
            )

        # hide a portion of the users with known labels
        known = [u for u in self.iter_users(selector)]

        print(f"no users with labels: {len(known)}/{len(self.users)}")
        test_users = random.choices(known, k=int(len(known) * (1 - train_size)))

        for user in self.iter_users():
            if user in test_users:
                user.add_meta("hidden", True)
                user.add_meta("dummy_label", None)

            else:
                # those who labels are not known OR will be used in training are still kept that way
                user.add_meta("hidden", False)
                user.add_meta("dummy_label", user.meta["label"])

            self.users[user.id] = user
        self.add_meta_field("hidden")
        return

    def add_meta_field(self, val: str):
        # Keep track of the metadata available for users in this collection
        self.meta.append(val)
        return

    def get_meta_fields(self):
        # Return the metadata available for users in this collection
        return self.meta

    def from_graph(self):
        # Construct population from graph df
        return

    def to_graph(self):
        # Return a dataframe representing graph
        # pandas DataFrame containing edges and metadata.
        # The first two columns of this DataFrame contain the source and target vertices for each edge
        return

    def get_user_meta(self):
        """
        Return all meta fields for users in this collection
        """
        meta_fields = []
        for user in self.users.values():
            for field in user.meta.keys():
                if field not in meta_fields:
                    meta_fields += [field]
        return meta_fields

    def get_all_user_meta(self, field: str = ""):
        """
        Return user-id meta for users in this collection
        """
        user_meta = {user.id: user.meta[field] for user in self.users.values()}
        return user_meta

    def get_user(self, user_id: str) -> User:
        """
        Gets User of the specified id from the Population

        :param user_id: id of User
        :return: User
        """
        return self.users[user_id]

    def random_user(self) -> User:
        """
        Get a random User from the Population

        :return: a random User
        """
        return random.choice(list(self.users.values()))

    def iter_users(
        self, selector: Optional[Callable[[User], bool]] = lambda user: True
    ) -> Generator[User, None, None]:
        """
        Get users in the Population, with an optional selector that filters for Users that should be included.

        :param selector: a (lambda) function that takes an User and returns True or False (i.e. include / exclude).
            By default, the selector includes all Users in the Population.
        :return: a generator of Users
        """
        for v in self.users.values():
            if selector(v):
                yield v

    def get_users_dataframe(
        obj, selector=lambda user: True, exclude_meta: bool = False
    ):
        """
        Get a DataFrame of the users of a given object with fields and metadata attributes,
        with an optional selector that filters for users that should be included.
        Edits to the DataFrame do not change the Population in any way.
        :param exclude_meta: whether to exclude metadata
        :param selector: a (lambda) function that takes a User and returns True or False (i.e. include / exclude).
            By default, the selector includes all Users that compose the object.
        :return: a pandas DataFrame
        """
        ds = dict()
        for user in obj.iter_users(selector):
            d = user.to_dict().copy()
            if not exclude_meta:
                for k, v in d["meta"].items():
                    d["meta." + k] = v
            del d["meta"]
            ds[user.id] = d

        df = pd.DataFrame(ds).T
        df = df.set_index("id")
        # df["speaker"] = df["speaker"].map(lambda spkr: spkr.id)
        meta_columns = [k for k in df.columns if k.startswith("meta.")]
        return df
        # return df[
        #     ["timestamp", "text", "speaker", "reply_to", "conversation_id"] + meta_columns + ["vectors"]
        # ]

    def filter_users_by(self, selector: Callable[[User], bool]):
        """
        Returns a new Population that includes only a subset of Users within this Population.
        Vectors are not preserved.

        :param selector: function for selecting which
        :return: a new Population with a subset of the Users
        """
        users = list(self.iter_users(selector))
        new_population = Population(users=users)
        for user in new_population.iter_users():
            user.meta.update(self.get_user(user.id).meta)
        return new_population

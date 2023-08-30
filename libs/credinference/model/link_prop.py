"""
- Locred:
    - Weighted PageRank where personalize vector is calculated for nodes in blacklist (untrustworthy)
    - Input: RT graph
    if personalized around bad nodes, score reflects LoCred - 
    if personalized around good nodes, score reflects reputation
- Personalized PageRank Trust (PPT): 
    - Weighted PageRank where personalize vector is calculated for nodes in whitelist (trustworthy)
    - Input: Trust-directed graph (reverse direction of RT graph)
- PageRank Trust (PT): PPT with no personalization
    - Input: Trust-directed graph (reverse direction of RT graph)
- Reputation scaling: Locred scaled by PPT scores
- TrustRank: 
    1. Uses Weighted PageRank for seed selection
    2. Calculates PPT scores, where personalization is based on both good and bad nodes calculated from step 1.
    - Input: Trust-directed graph (reverse direction of RT graph)
"""
import igraph
from credinference.util import get_dict_val, normalize

METHODS = {
    "locred": {"personalized": "bad"},
    "ppt": {"personalized": "good"},
    "pt": {"personalized": None},
    "trustrank": {"personalized": "good",},
}
# mapping between account type for personalization and label value
personalization_key = {"bad": 1, "good": 0}


def get_named_edge_dataframe(graph):
    # Returns a dataframe of the edges in graph, where source and target are node names instead of indices
    df = graph.get_edge_dataframe()
    df_vert = graph.get_vertex_dataframe()
    df["source"].replace(df_vert["name"], inplace=True)
    df["target"].replace(df_vert["name"], inplace=True)
    # df_vert.set_index('name', inplace=True)  # Optional
    return df


def get_inverse_graph(graph):
    """
    Return an inverse graph G' where the adjacency matrix G' = G.T (transpose of original graph)
    """
    df = graph.get_edge_dataframe()
    df_vert = graph.get_vertex_dataframe()
    inverse = igraph.Graph.DataFrame(
        edges=df[["target", "source", "weight"]], vertices=df_vert, directed=True
    )
    assert inverse.vs["name"] == graph.vs["name"]
    return inverse


def locred(graph, **kwargs) -> dict():
    """
    Returns LoCred score of the nodes in a graph using igraph pagerank method.
    - graph (igraph.Graph): RT network where direction is info spread (retweeter -> retweeted)
    """
    label = get_dict_val(METHODS, ["locred", "personalized"])
    p_label = personalization_key[label]
    return wppr(graph, score_name="locred", p_label=p_label, **kwargs)


def pagerank_trust(graph, personalized: bool = False, **kwargs) -> dict():
    """
    Returns PageRank Trust (PT) scores
    - graph (igraph.Graph): RT network where direction is info spread (retweeter -> retweeted)
    - personalized (bool): if True, return Personalized PageRank Trust (PPT) scores,
        where personalize vector is calculated for nodes in whitelist (trustworthy)

    """
    # convert graph to Trust-directed graph (reverse direction of RT graph)
    trust_graph = get_inverse_graph(graph)
    if personalized is True:
        label = get_dict_val(METHODS, ["ppt", "personalized"])
        p_label = personalization_key[label]
        return wppr(trust_graph, score_name="ppt", p_label=p_label, **kwargs)
    else:
        return wppr(trust_graph, score_name="pt", p_label=None, **kwargs)


def reputation_scaling(graph, alpha1: float, alpha2: float, **kwargs) -> dict():
    """
    Return Reputation Scaling scores, where Locred scores are scaled by PPT scores
    - graph (igraph.Graph): RT network where direction is info spread (retweeter -> retweeted)
    - alpha1: jumping factor in wpr calculation of trust
    - alpha2: jumping factor in wpr calculation of misinfo spreading
    """
    trust_graph = get_inverse_graph(graph)
    ppt_scores = pagerank_trust(trust_graph, personalized=True, alpha=alpha1, **kwargs)

    locred_scores = locred(graph, alpha=alpha2, **kwargs)
    # scale results:
    rs_scores = dict()
    for node, t_i in ppt_scores.items():
        rs_scores[node] = t_i * (1 - locred_scores[node])
    return rs_scores


def trustrank(
    graph,
    alpha1: float,
    alpha2: float,
    num_seeds: int,
    seed_selection: str = "pr",
    **kwargs,
) -> dict():
    """
    Return TrustRank scores
    1. Uses Weighted PageRank for seed selection
    2. Calculates PPT scores, where personalization is based on both good and bad nodes calculated from step 1.
    - graph (igraph.Graph): RT network where direction is info spread (retweeter -> retweeted)
    - alpha1: initial wpr calculation to select seeds
    - alpha2: final reputation calculation
    - seed_selection: {'pr','inverse_pr'}
    """
    trust_graph = get_inverse_graph(graph)
    if seed_selection == "pr":
        pt = pagerank_trust(trust_graph, personalized=False, alpha=alpha1)
    elif seed_selection == "inverse_pr":
        # perform pr on inversed graph
        pt = pagerank_trust(graph, personalized=False, alpha=alpha1)

    topn = sorted(pt, key=pt.get, reverse=True)[:num_seeds]
    seeds = [v for v in trust_graph.vs if v["name"] in topn]
    label = get_dict_val(METHODS, ["trustrank", "personalized"])
    p_label = personalization_key[label]

    return wppr(trust_graph, score_name="ppt", p_label=p_label, seeds=seeds, **kwargs)


## TODO: handle p_label=None
def wppr(
    graph,
    score_name: str = "locred",
    p_label: int = 1,
    seeds: list = None,
    weight_col: str = "weight",
    alpha: float = 0.85,
) -> dict():
    """
    Returns Weighted Personalized PageRank score of the nodes

    Inputs:
        - graph (igraph.Graph): weighted RT network where nodes have an attribute called 'label' (0:good, 1: bad, -1:nan)
        - seeds (list): list of node indices to use for personalization (only apply to trustrank)
        - params: alpha: jumping parameter; niter: max iteration, eps: convergence criteria (niter, eps deprecated in latest igraph version)
        - p_label (int): criteria for personalization (to reset the random walk with high probs at bad or good nodes)

    Outputs:
        - scores: (dict): {uid: score}
    """
    try:
        print(f"Calculating {score_name} (alpha={alpha}) ..")

        if (p_label is not None) and (graph.vs[0]["label"] is None):
            # if method is based on personalization (not pt) and labels are not available, raise error
            raise ValueError(
                "Nodes in the graph need to be labeled to run personalized algorithms"
            )
        if score_name == "pt":
            N = graph.vcount()
            personalization = [1 / N] * N

        elif score_name == "trustrank":
            bad_nodes = [
                v for v in graph.vs if v["label"] == personalization_key["bad"]
            ]
            good_nodes = [
                v for v in graph.vs if v["label"] == personalization_key["good"]
            ]
            bias_vec = []
            for v in graph.vs:
                if (v in seeds) and (v in bad_nodes):
                    bias_vec.append(0)
                if (v in seeds) and (v in good_nodes):
                    bias_vec.append(1)
                else:
                    bias_vec.append(0.5)
            personalization = normalize(bias_vec)
            print("Number of labeled nodes: ", len(bad_nodes) + len(good_nodes))
        else:
            # create personalization vector
            labeled_nodes = [v for v in graph.vs if v["label"] == p_label]
            print("Number of labeled nodes: ", len(labeled_nodes))
            personalization = [
                1 / len(labeled_nodes) if v in labeled_nodes else 0 for v in graph.vs
            ]

        # compute scores. vertices=None: get scores for all vertices
        # niter: max number of iterations used in power method. Ignored if implementation doesn't use power method.
        # eps: difference between scores between iteration for stopping
        results = graph.personalized_pagerank(
            vertices=None,
            directed=True,
            damping=alpha,
            reset=personalization,
            weights=weight_col,
        )

        # add scores as node score_name to graph (score_name name is 'locred' or 'reputation')
        graph.vs[score_name] = results
        scores = {v["name"]: v[score_name] for v in graph.vs}

    except Exception as e:
        print(e)
        print(f"Could not perform {score_name}")
        return
    return scores

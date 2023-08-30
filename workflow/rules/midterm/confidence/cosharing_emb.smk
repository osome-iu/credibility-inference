"""
Run graph emb classification on cosharing network 
Vary confidence level. We already know the optimal config for node2vec.
"""

direction = "undirected"
embedding="fast"
p_vals=[1]
q_vals = [1]
confidences = [0.5, 0.6, 0.7, 0.8, 0.9, 1]

DATASET = "midterm"
network = "cosharing"

ABS_PATH = "/N/slate/baotruon/infopolluters"

label_dir = os.path.join(ABS_PATH, "data", DATASET, "labels")
edgelist_dir = os.path.join(ABS_PATH, "data", DATASET, "edgelists")

derived = os.path.join(ABS_PATH, "data", DATASET, "node2vec", network)
#predictions 
res_dir = os.path.join(ABS_PATH, "results", DATASET, "node2vec", network)

edgelist_fname = "cosharing_edgelist__115461"
print(os.getcwd())

rule all:
    input:
        expand(os.path.join(res_dir, f"confidence{{conf}}_p{{p}}q{{q}}__{edgelist_fname}.pkl"), p=p_vals, q=q_vals, conf=confidences)
        # expand(os.path.join(res_dir, f"{{direction}}/p{{param['p']}}q{{param['q']}}__{edgelist_fname}.pkl"),direction=directions, param=params) # we could do this but it's ugly

rule clf:
    input:
        graph_file=os.path.join(edgelist_dir, f"{edgelist_fname}.txt"),
        embedding=ancient(os.path.join(derived, f"{embedding}_{direction}_p{{p}}q{{q}}__{edgelist_fname}.model")),
        labels = os.path.join(label_dir, "user_info.parquet")
    output: os.path.join(res_dir, f"confidence{{conf}}_p{{p}}q{{q}}__{edgelist_fname}.pkl")
    shell: """
    python3 workflow/scripts/emb_eval.py -i {input.graph_file} -e {input.embedding} -l {input.labels} -o {output} --confidence {wildcards.conf}
    """

## Elio runs into memory error so we're using fastnode2vec by default for cosharing network
rule embed:
    input: 
        graph_file = os.path.join(edgelist_dir, f"{edgelist_fname}.txt"),
        config = ancient(os.path.join(derived, "config",  "config__p{p}q{q}.json")),
    output:os.path.join(derived, f"{embedding}_{direction}_p{{p}}q{{q}}__{edgelist_fname}.model")
    shell: """
    python3  workflow/scripts/get_graph_emb.py -i {input.graph_file} -c {input.config} -o {output} -e {embedding} -d {direction}
    """

rule config:
    output:os.path.join(derived, "config",  "config__p{p}q{q}.json")
    shell: """
    python3  workflow/scripts/config_graphemb.py -p {wildcards.p} -q {wildcards.q} -o {output}
    """

# rule make_edgelist:
#     input: os.path.join(edgelist_dir, "bipartite_edgelist.parquet")
#     output: os.path.join(edgelist_dir, f"{edgelist_fname}.parquet")
#     shell: """
#     python3 workflow/preprocess/coshare_projection.py {input} {output}
#     """
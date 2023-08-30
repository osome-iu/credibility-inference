"""
Run graph emb classification on RT network 
Vary confidence level. We already know the optimal config for node2vec.
"""

directions = ["undirected", "directed", "trust"]
p_vals=[1, 2, 2]
q_vals = [1, 1, 0.5]
confidences = [0.5, 0.6, 0.7, 0.8, 0.9, 1]

DATASET = "midterm"
network = "rt"

ABS_PATH = "/N/slate/baotruon/infopolluters"

label_dir = os.path.join(ABS_PATH, "data", DATASET, "labels")
edgelist_dir = os.path.join(ABS_PATH, "data", DATASET, "edgelists")

derived_dir = os.path.join(ABS_PATH, "data", DATASET, "node2vec", network)
#predictions 
res_dir = os.path.join(ABS_PATH, "results", DATASET, "node2vec", network)

edgelist_fname = "rt_edgelist__410914"
print(os.getcwd())


rule all:
    input:
        expand(expand(os.path.join(res_dir, f"{{direction}}/confidence{{{{conf}}}}_p{{p}}q{{q}}__{edgelist_fname}.pkl"),zip, direction=directions, p=p_vals, q=q_vals), conf=confidences)


rule clf:
    input:
        graph_file=os.path.join(edgelist_dir, f"{edgelist_fname}.txt"),
        embedding=ancient(os.path.join(derived_dir, f"{{direction}}/p{{p}}q{{q}}__{edgelist_fname}.model")),
        labels = os.path.join(label_dir, "user_info.parquet")
    output:os.path.join(res_dir, f"{{direction}}/confidence{{conf}}_p{{p}}q{{q}}__{edgelist_fname}.pkl")
    shell: """
    python3 workflow/scripts/emb_eval.py -i {input.graph_file} -e {input.embedding} -l {input.labels} -o {output} --confidence {wildcards.conf}
    """

rule embed:
    input: 
        graph_file = os.path.join(edgelist_dir, f"{edgelist_fname}.txt"),
        config = ancient(os.path.join(derived_dir, "config",  "config__p{p}q{q}.json")),
    output:os.path.join(derived_dir, f"{{direction}}/p{{p}}q{{q}}__{edgelist_fname}.model")
    shell: """
    python3  workflow/preprocess/get_graph_emb.py -i {input.graph_file} -c {input.config} -o {output} -d {wildcards.direction}
    """

rule config:
    output:os.path.join(derived_dir, "config",  "config__p{p}q{q}.json")
    shell: """
    python3  workflow/scripts/config_graphemb.py -p {wildcards.p} -q {wildcards.q} -o {output}
    """
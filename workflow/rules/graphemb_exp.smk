"""
Do classification on different types of embedding (varying packages and directionality)
To investigate effect of directionality
"""

directions = ["undirected", "directed", "trust"]
embedding_types = ["elio", "fast"]
p_vals=[0.25, 0.5,1,2,4]
q_vals = [0.25, 0.5,1,2,4]

data_dir = "/N/slate/baotruon/infopolluters/data"
intermediate_dir = os.path.join(data_dir, "node2vec")
# res_dir = "/N/u/baotruon/Carbonate/infopolluters/results"
res_dir = "/N/slate/baotruon/infopolluters/results_04112023/node2vec"
print(os.getcwd())
rule all:
    input:
        expand(os.path.join(res_dir, "{direction}/{emb}_p{p}q{q}__rt_cc_size322208.parquet"),direction=directions,emb=embedding_types, p=p_vals, q=q_vals),


rule clf:
    input:
        graph_file=os.path.join(data_dir, "rt_cc_size322208.txt"),
        embedding=ancient(os.path.join(intermediate_dir, "{direction}/{emb}_p{p}q{q}__rt_cc_size322208.model")),
        labels = os.path.join(data_dir, "user_labels.csv")
    output:os.path.join(res_dir, "{direction}/{emb}_p{p}q{q}__rt_cc_size322208.parquet")
    shell: """
    python3 workflow/scripts/graphemb_eval.py -i {input.graph_file} -e {input.embedding} -l {input.labels} -o "{res_dir}/{wildcards.direction}" --mode preds
    """

rule embed:
    input: 
        graph_file = os.path.join(data_dir, "rt_cc_size322208.txt"),
        config = ancient(os.path.join(intermediate_dir, "config",  "config__p{p}q{q}.json")),
    output:os.path.join(intermediate_dir, "{direction}/{emb}_p{p}q{q}__rt_cc_size322208.model")
    shell: """
    python3  workflow/scripts/get_graph_emb.py -i {input.graph_file} -o {output} -c {input.config} -e {wildcards.emb} -d {wildcards.direction}
    """

rule config:
    output:os.path.join(intermediate_dir, "config",  "config__p{p}q{q}.json")
    shell: """
    python3  workflow/scripts/config_graphemb.py -p {wildcards.p} -q {wildcards.q} -o {output}
    """

"""
Eval classification on different graph centrality measure ranking
vary confidence level
Include the data preprocessing steps 
"""
import numpy
import glob 

# methods = ["CoCred", "HITS"]
methods = ["CoCred", "HITS", "CoHITS",  "BGRM", "BiRank"]
confidences = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
DATASET = "covid"

ABS_PATH = "/N/slate/baotruon/infopolluters"
url_dir = os.path.join(ABS_PATH, "data", DATASET, "urls")

# ABS_PATH = "/Users/baott/infopolluters"
# url_dir = "/data_volume/midterm2022/intermediate_files/entities/twitter/expanded_urls"
label_dir = os.path.join(ABS_PATH, "data", DATASET, "labels")
# # raw .json.gz
# raw_data_dir = os.path.join(ABS_PATH, "data", DATASET, "raw")
# # intermediate (processed df, edgelist, labels etc. ßßßßß)
# processed_dir = os.path.join(ABS_PATH, "data", DATASET, "processed")

edgelist_dir = os.path.join(ABS_PATH, "data", DATASET, "edgelists")

#predictions 
res_dir =os.path.join(ABS_PATH, "results", DATASET, "bipartite")

edgelist_fname = "bipartite_edgelist"
# print(os.getcwd())

# raw_tweet_fnames = [os.path.basename(fpath).replace(".json.gz", "") for fpath in glob.glob(f"{raw_data_dir}/*.json.gz")]
# print(raw_tweet_fnames)
rule all:
    input:
        expand(os.path.join(res_dir, "{method}", f"confidence{{confidence}}__{edgelist_fname}.pkl"),method=methods,confidence=confidences),

rule clf:
    input:
        graph=os.path.join(edgelist_dir, f"{edgelist_fname}.parquet"),
        user_labels = os.path.join(label_dir, "user_info.parquet")
    output:os.path.join(res_dir, "{method}", f"confidence{{confidence}}__{edgelist_fname}.pkl")
    shell: """
    python3 workflow/scripts/cocred_eval.py -i {input.graph} -u {input.user_labels} -o {output} --method {wildcards.method} --confidence {wildcards.confidence}
    """

rule user_labels:
    input:
        platforms = os.path.join(label_dir, "platform.csv"),
        domain_labels = os.path.join(label_dir, "domain_labels.csv")
    output:
        domain_idx = os.path.join(label_dir, "domain_idx.csv"),
        user_domain = os.path.join(label_dir, "user_domains.pkl"),
        user_labels = os.path.join(label_dir, "user_info.parquet")
    shell:"""
    python3 workflow/preprocess/user_shared_domains.py -i {url_dir} -d {input.domain_labels} --platforms {input.platforms} --outpkl {output.user_domain} --outcsv {output.domain_idx} --outparquet {output.user_labels}
    """

rule make_edgelist:
    input: 
        input_dir = url_dir,
        platforms = os.path.join(label_dir, "platform.csv"),
    output: os.path.join(edgelist_dir, f"{edgelist_fname}.parquet")
    shell: """
    python3 workflow/preprocess/bipartite_edgelist.py {input.input_dir} {input.platforms} {output}
    """

# rule expand_urls:
#     input: os.path.join(processed_dir, "{raw_fname}.parquet")
#     output: os.path.join(url_dir, "{raw_fname}.parquet")
#     shell: """
#     python3 workflow/preprocess/expand_url.py {input} {output}
#     """

# rule preprocess_tweets:
#     input: os.path.join(raw_data_dir, "{raw_fname}.json.gz")
#     output: os.path.join(processed_dir, "{raw_fname}.parquet")
#     shell: """
#     python3 workflow/preprocess/extract_tweets.py {input} {output}
#     """
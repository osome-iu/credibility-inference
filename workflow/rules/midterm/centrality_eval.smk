"""
Last updated: June 1, 2023
Perform eval on user splits
Do classification on different graph centrality measure ranking
vary alpha
Include the data preprocessing steps 
"""
import numpy
import glob 

methods = ["locred", "reputation_scaling", "ppt", "pt", "trustrank"]
# alphas = [numpy.round(a, 2) for a in numpy.arange(0,1.1, step=0.1)]
alphas = [0.8]
DATASET = "midterm"

ABS_PATH = "/data_volume/playground/baotruon/infopoluters"
# # local testing 
# ABS_PATH = "/Users/baott/infopolluters"
label_dir = os.path.join(ABS_PATH, "data", DATASET, "labels")
# raw .json.gz
raw_data_dir = os.path.join(ABS_PATH, "data", DATASET, "raw")
# intermediate (processed df, edgelist, labels etc. )
processed_dir = os.path.join(ABS_PATH, "data", DATASET, "processed")
edgelist_dir = os.path.join(ABS_PATH, "data", DATASET, "edgelists")

#predictions 
res_dir = os.path.join(ABS_PATH, "results", DATASET, "centrality")

edgelist_fname = "rt_edgelist"
# print(os.getcwd())

# raw_tweet_fnames = [os.path.basename(fpath).replace(".json.gz", "") for fpath in glob.glob(f"{raw_data_dir}/*.json.gz")]
# print(raw_tweet_fnames)
rule all:
    input:
        expand(os.path.join(res_dir, "{method}", f"alpha{{alpha}}__{edgelist_fname}.pkl"),method=methods,alpha=alphas),

rule clf:
    input:
        graph_file=os.path.join(edgelist_dir, f"{edgelist_fname}.parquet"),
        labels = os.path.join(label_dir, "user_info.parquet")
    output:os.path.join(res_dir, "{method}", f"alpha{{alpha}}__{edgelist_fname}.pkl")
    shell: """
    python3 workflow/scripts/centrality_eval.py -i {input.graph_file} -l {input.labels} -o {output} --method {wildcards.method} -a {wildcards.alpha}
    """

# rule make_edgelist:
#     input: 
#         input_dir = processed_dir,
#         files = expand(os.path.join(processed_dir, "{raw_fname}.parquet"), raw_fname = raw_tweet_fnames),
#     output: os.path.join(edgelist_dir, f"{edgelist_fname}.parquet")
#     shell: """
#     python3 workflow/preprocess/rt_edgelist.py {input.input_dir} {output}
#     """

# rule preprocess_tweets:
#     input: os.path.join(raw_data_dir, "{raw_fname}.json.gz")
#     output: os.path.join(processed_dir, "{raw_fname}.parquet")
#     shell: """
#     python3 workflow/preprocess/extract_tweets.py {input} {output}
#     """
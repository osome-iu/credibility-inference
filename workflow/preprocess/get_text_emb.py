"""
Purpose:
    Create mapping of user_id - text embedding vector
Inputs:
    - pretrained embedding (GloVe or fasttext)
    - .parquet file of users and shared posts 
NOTE: Each df has at least the following columns: user_id, tweet_type, text
Outputs: 
    .pkl file of embedding (dict): {node_name: embedding_vector}
"""
from infopolluter import get_text2vec_mapping, load_glove_embedding
from infopolluter.util import get_account_labels, preprocess_text
import pandas as pd
import os
import pickle
import argparse
import json
import sys
from tqdm import tqdm


def clean_text(df):
    """
    Clean text in a dataframe
    df with ['user_id', 'post_id', 'text'] columns
    Use this function instead of df.apply because apply is vey slow
    """
    # created df of clean text
    text_records = df[["post_id", "text"]].to_dict("records")
    cleaned = []
    for record in tqdm(text_records, desc="Cleaning text in df"):
        rec = record.copy()
        rec["cleaned_text"] = preprocess_text(record["text"])
        cleaned.append(rec)
    cleaned_df = pd.DataFrame.from_records(cleaned)

    # merge with original df
    data = pd.merge(df[["user_id", "post_id"]], cleaned_df, on="post_id")
    return data


def read_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--embedding",
        type=str,
        help=".txt file path for pretrained embedding (GloVe or fasttext)",
    )
    parser.add_argument(
        "-t",
        "--textfile",
        type=str,
        help="File path for .parquet file of user and associated posts",
    )

    parser.add_argument(
        "-o",
        "--outfile",
        type=str,
        help=".pkl file of graph embedding (dict of user_id-vector)",
    )
    args = parser.parse_args(args)
    return args


if __name__ == "__main__":
    args = read_args(sys.argv[1:])

    # embedding_path = "data/glove/glove.twitter.27B/glove.twitter.27B.200d.txt"
    # data_path = "data/covid/urls/posts.parquet"
    # outfile = "data/covid/derived/posts_glove200d.pkl"
    embedding_path = args.embedding
    data_path = args.textfile
    outfile = args.outfile

    print("Create mapping of user_id to embedding vector.. ")
    try:
        df = pd.read_parquet(data_path)
        if any([i not in df.columns for i in ["user_id", "text"]]):
            raise ValueError("Post df is required to have 'user_id' and 'text' column")

        ## clean text
        print("Cleaning raw text.. ")
        # # TODO: improve embedding by including hashtags during cleaning
        # df["cleaned_text"] = df["text"].apply(lambda x: preprocess_text(x))

        df = clean_text(df)
        # remove non-english texts
        df = df[~df["cleaned_text"].isna()].reset_index()

        ## aggregate a user's text and map to a vector
        w2v_model = load_glove_embedding(embedding_path)
        embeddings = get_text2vec_mapping(df, w2v_model)
        print("Done. Saving..")

        pickle.dump(embeddings, open(outfile, "wb"))

        # fname = os.path.basename(filename)
        # df_out = os.path.join(args.outpath, f"{fname}.parquet")
        # report_out = os.path.join(args.outpath, f"{fname}.pkl")
        # pickle.dump(report, open(report_out, "wb"))
        # df.to_parquet(df_out, engine="pyarrow")

    except Exception as e:
        print(e)
        pass


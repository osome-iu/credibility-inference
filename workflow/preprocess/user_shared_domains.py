"""
    Combined all parsed df of tweets from a period of time.
    Index domain labels (starts from 0)
    Make User shared domains dict, where list of domains is a list of domain indices matching 
    - Input: 
        - Directory of .json.gz files of parsed tweets (.parquet). Each row is a tweet
        - Domain labels: df of domain labels (e.g: Newsguard)  
            .csv (sep=',) ; need to have at least ['Rating', 'domain'] column. 
        - Platform list: .csv (sep=',) of platform names
    - Args: 
        start, end (all inclusive, optional): if files are named by date, only use files from the period specified
    NOTE: Each df has at least the following columns: user_id, domain
    - Output: 
        - (.pkl file) of User shared domains (dict: user_id - list of domains)
        - (.csv file) of indexed domain labels
"""

import sys
import pandas as pd
import glob
from infopolluter import get_logger
import os
import pickle
from tqdm import tqdm
import argparse
import fnmatch
from infopolluter.util import get_score, user_col, DOMAIN_COL


def parse_args(args):
    parser = argparse.ArgumentParser(description="Get graph embedding",)
    parser.add_argument(
        "-i",
        "--indir",
        type=str,
        required=True,
        help="Directory to .parquet files of parsed tweets",
    )
    parser.add_argument(
        "-d",
        "--domainlabels",
        type=str,
        required=True,
        help=".csv file of domain labels",
    )

    parser.add_argument(
        "--platforms", type=str, required=True, help=".csv file of platform domains",
    )
    parser.add_argument(
        "--outcsv",
        type=str,
        required=True,
        help=".csv fpath to domain and index mapping",
    )
    parser.add_argument(
        "--outpkl",
        type=str,
        required=True,
        help=".pkl fpath to dict of user-shared domains",
    )
    parser.add_argument(
        "--outparquet",
        type=str,
        required=True,
        help=".parquet fpath to a df of user info (user_score, confidence, user_label)",
    )
    parser.add_argument(
        "-s", type=str, help="start date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "-e", type=str, help="end date in YYYY-MM-DD format",
    )
    print(os.getcwd())
    args = parser.parse_args(args)
    return args


if __name__ == "__main__":
    ## LOGGING
    log_dir = "logs"

    logger = get_logger(
        log_dir=log_dir,
        full_log_path=os.path.join(log_dir, f"user_shared_domains.log"),
        also_print=True,
        tqdm=True,
    )
    args = parse_args(sys.argv[1:])
    ## READ ARGS
    in_path = args.indir
    label_path = args.domainlabels
    platform_path = args.platforms
    out_pkl = args.outpkl
    out_csv = args.outcsv
    out_parquet = args.outparquet
    start = args.s
    end = args.e

    REQUIRED_COLS = [user_col, DOMAIN_COL]
    file_names = glob.glob(f"{in_path}/*.parquet")

    ## Optional: only extract from a period of time
    if (start is not None) and (end is not None):
        logger.info(f"Subsetting files from {start} to {end}")
        dates = pd.date_range(start=start, end=end)
        input_files = []
        for date in dates:
            date_string = date.strftime("%Y-%m-%d")
            # fnmatch.filter always returns a list
            input_files.extend(fnmatch.filter(file_names, f"*{date_string}*"))

        file_names = input_files

    # combine all dfs
    logger.info(f"Combining files (total {len(file_names)} files)")
    dfs = []
    for fpath in tqdm(file_names, desc="Combining posts"):
        try:
            raw_df = pd.read_parquet(fpath)
            if not all(col in raw_df.columns for col in REQUIRED_COLS):
                raise ValueError(
                    f"Expect post dataframe to contain these columns: {REQUIRED_COLS}"
                )
        except Exception as e:
            logger.debug(f"Skipping file {fpath}")
            continue

        dfs.append(raw_df)

    # all posts
    posts = pd.concat(dfs)

    domains = pd.read_csv(label_path)
    df = pd.merge(posts, domains, on="domain", how="left")

    platform_df = pd.read_csv(platform_path)
    platforms = set(platform_df["platform"].values)
    # exclude platforms
    df = df[~(df["domain"].isin(platforms))]
    df = df[~(df["Rating"] == "P")]
    # exclude satire
    df = df[~(df["Rating"] == "S")]
    df = df[["user_id", "domain"]]

    ## INDEX DOMAIN AND USERS
    domain_ids = pd.DataFrame(df["domain"].unique(), columns=["domain"]).reset_index()
    domain_ids = domain_ids.rename(columns={"index": "domain_index"})
    domain_ids = domain_ids.astype({"domain_index": int})
    df = df.merge(domain_ids, on="domain", how="outer")

    user_ids = pd.DataFrame(df["user_id"].unique(), columns=["user_id"]).reset_index()
    user_ids = user_ids.rename(columns={"index": "user_index"})
    user_ids = user_ids.astype({"user_index": int})
    df = df.merge(user_ids, on="user_id", how="outer")

    domain_labels = domain_ids.merge(domains, on="domain", how="left").sort_values(
        by="domain_index"
    )
    domain_labels.to_csv(out_csv, index=False)
    logger.info(f"Finish saving indexed domain labels to {out_csv}")

    ## Make User shared domains dict
    # assert len(df[df["domain_index"].isna()]) == 0
    print("cols: ", df.columns)
    ## NOTE: Slicing with numpy is much faster than iloc
    user_shared_domains = dict()
    user_info = []

    data = df[["domain_index", "user_index"]].values
    domain_scores = domain_labels["Score"].values
    for idx, row in tqdm(
        user_ids.iterrows(), desc="Creating dict of user-shared domains"
    ):
        # Get user shared domains
        user = row["user_index"]
        mask = data[:, 1] == user
        domain_idxs = data[mask, 0]
        user_shared_domains[row["user_id"]] = list(domain_idxs)

        # Get user scores
        try:
            score, known_frac = get_score(list(domain_idxs), domain_scores)
        except:
            score, known_frac = -1, 0
            continue

        info_row = {
            user_col: row["user_id"],
            "user_score": score,
            "confidence": known_frac,
        }
        user_info.append(info_row)

    user_df = pd.DataFrame.from_records(user_info)
    user_df.to_parquet(out_parquet, engine="pyarrow")

    with open(out_pkl, "wb") as f:
        pickle.dump(user_shared_domains, f)

    logger.info(f"Finish saving User-domain shared dict to {out_pkl}")

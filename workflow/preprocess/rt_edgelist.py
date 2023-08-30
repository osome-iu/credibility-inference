"""
    Combined all parsed df of tweets from a period of time into a Retweet edgelist
    - Input: Directory of .json.gz files of parsed tweets (.parquet). Each row is a tweet
    NOTE: 
        - Input dir is created with `preprocess/extract_tweets`
        - If the tweet is a retweet, the associated URL is the same as the URL shared in the original tweet
        - Each df has at least the following columns: user_id, retweeted_user_id, tweet_type
    - Args: 
        start, end (all inclusive, optional): if files are named by date, only use files from the period specified
        minshare (optional): minimum number of shares to filter users by (default: 5)
    - Output: .txt file of Weighted RT edgelist (user - user - tweet count)
        Edge direction: from retweeter to the one being retweeted 
"""

# TODO: Previously we only look at users who shared at least 1 domain
# (the same data for RT net as Bipartite net) -- Might want to change this later?
import sys
import pandas as pd
import glob
from infopolluter import get_logger
import os
import fnmatch
import argparse


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Make RT edgelist from tweets with shared domains",
    )
    parser.add_argument(
        "-i",
        "--indir",
        type=str,
        required=True,
        help="Directory to .parquet files of parsed tweets",
    )
    parser.add_argument(
        "-o",
        "--outpath",
        type=str,
        required=True,
        help=".parquet fpath to weighted RT edgelist",
    )
    parser.add_argument(
        "-s", type=str, help="start date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "-e", type=str, help="end date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--minshare",
        type=int,
        help="minimum number of shares to filter users by (default 5 if not provided)",
    )
    print(os.getcwd())
    args = parser.parse_args(args)
    return args


if __name__ == "__main__":
    ## LOGGING
    log_dir = "logs"

    logger = get_logger(
        log_dir=log_dir,
        full_log_path=os.path.join(log_dir, f"rt_edgelist.log"),
        also_print=True,
    )

    ## READ ARGS
    args = parse_args(sys.argv[1:])
    in_path = args.indir
    out_path = args.outpath
    start = args.s
    end = args.e
    ## Optional: filter users who share less than n times
    MIN_SHARES = args.minshare if args.minshare is not None else 0

    REQUIRED_COLS = ["user_id", "retweeted_user_id", "tweet_type"]
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
    for fpath in file_names:
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

    df = pd.concat(dfs)

    raw_no_users = len(df)
    ## FILTER USERS
    users = df.groupby(["user_id"]).tweet_type.count().reset_index()
    user_to_filter = users[users.tweet_type <= MIN_SHARES]["user_id"].values
    df = df[~df["user_id"].isin(user_to_filter)]
    logger.info(
        f"Number of retweets after removing users with less than {MIN_SHARES}: {len(df)}/{raw_no_users}"
    )

    ## MAKE RETWEET EDGELISTs
    rt = df[df["tweet_type"] == "retweet"][REQUIRED_COLS]
    rt = (
        rt.groupby(["user_id", "retweeted_user_id"])["tweet_type"].count().reset_index()
    )

    rt = rt.rename(columns={"tweet_type": "weight"})
    rt = rt.astype({"user_id": str, "retweeted_user_id": str})

    logger.info(f"Number of unique edges: {len(rt)}/{len(df)}")

    # make sure column order is correct (important for centrality-based measures on network later)
    rt = rt[["retweeted_user_id", "user_id", "weight"]]
    # save edgelist
    rt.to_csv(
        out_path.replace(".parquet", ".txt"),
        sep=" ",
        encoding="utf-8",
        index=False,
        header=False,
    )
    rt.to_parquet(out_path, engine="pyarrow")
    logger.info(f"--- Finished writing RT edgelist to {out_path} ---")

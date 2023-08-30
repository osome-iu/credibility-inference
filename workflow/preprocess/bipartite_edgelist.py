"""
    Combined all parsed df of tweets from a period of time into a Bipartite edgelist.
    Exclude domains that are platforms by default.
    Inputs: 
        - in_path: Directory of .parquet files of parsed tweets. Each row is a tweet
            NOTE: Each df has at least the following columns: user_id, domain, tweet_type
        - platform_fpath: .csv file of domains that are platforms (fb, twitter, etc.)
    - Args: 
        start, end (all inclusive, optional): if files are named by date, only use files from the period specified
        minshare (optional): minimum number of shares to filter users and domains by (default: 5)
    Output: .txt file of Weighted Bipartite edgelist (user - domain - number of shares)
"""
import pandas as pd
import os
import sys
import glob
from infopolluter import get_logger
import fnmatch

if __name__ == "__main__":
    ## LOGGING
    log_dir = "logs"

    logger = get_logger(
        log_dir=log_dir,
        full_log_path=os.path.join(log_dir, f"bipartite_edgelist.log"),
        also_print=True,
    )

    ## READ ARGS
    in_path = sys.argv[1]
    platform_fpath = sys.argv[2]
    out_path = sys.argv[3]

    # in_path = "data/covaxxy/urls"
    # platform_fpath = "data/domain_labels/platform.csv"
    # out_path = "bipartite.parquet"
    # combine all dfs
    MIN_SHARES = 5
    REQUIRED_COLS = ["user_id", "domain", "tweet_type"]
    file_names = glob.glob(f"{in_path}/*.parquet")

    ## Optional: only extract from a period of time
    if len(sys.argv) > 4:
        start = sys.argv[4]
        end = sys.argv[5]
        logger.info(f"Subsetting files from {start} to {end}")
        dates = pd.date_range(start=start, end=end)
        input_files = []
        for date in dates:
            date_string = date.strftime("%Y-%m-%d")
            # fnmatch.filter always returns a list
            input_files.extend(fnmatch.filter(file_names, f"*{date_string}*"))

        file_names = input_files

    ## Optional: filter domains that appear less than n times and users who share less than n times
    if len(sys.argv) > 6:
        MIN_SHARES = int(sys.argv[6])

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

    raw = pd.concat(dfs)

    ## MAKE BIPARTITE EDGELIST
    platform_df = pd.read_csv(platform_fpath)
    platforms = set(platform_df["platform"].values)

    # exclude_platforms
    df = raw[~(raw["domain"].isin(platforms))]
    logger.info(
        f"Df after removing users who shared platform domains: {len(df)}/{len(raw)}"
    )

    ## filter domains that appear less than 10 times, users who share less than 10 times
    users = df.groupby(["user_id"]).tweet_type.count().reset_index()
    user_to_filter = users[users.tweet_type <= MIN_SHARES]["user_id"].values

    domains = df.groupby(["domain"]).tweet_type.count().reset_index()
    domain_to_filter = domains[domains.tweet_type <= MIN_SHARES]["domain"].values

    df = df[~(df["domain"].isin(domain_to_filter) | df["user_id"].isin(user_to_filter))]
    logger.info(
        f"Df after removing users and domains with less than {MIN_SHARES}: {len(df)}/{len(raw)}"
    )

    # Bipartite user-domain df
    bipartite = df.groupby(["user_id", "domain"])["tweet_type"].count().reset_index()
    bipartite = bipartite.astype({"user_id": str})
    bipartite = bipartite.rename(columns={"tweet_type": "weight"})

    logger.info(f"Bipartite edges: {len(bipartite)}")
    # save edgelist
    if len(bipartite) > 0:
        bipartite.to_parquet(out_path, engine="pyarrow")
        # bipartite.to_csv(
        #     out_path.replace(".parquet", ".txt"), sep=" ", encoding="utf-8", index=False
        # )
        logger.info(f"--- Finished writing Bipartite edgelist to {out_path} ---")
    else:
        with open(out_path, "a") as f:
            pass

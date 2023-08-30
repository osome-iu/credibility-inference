"""
Purpose
    Expand the shortened URLs. 
    If dates are not specified, expand yesterday's file. 
    If platform are not specified, expand all platforms 

Inputs:
    - .parquet file with a column 'raw_url' to expand

Output:
    - .parquet file with the same content of the input file but
        1. Those records without raw_url are removed
        2. There is two extra columns raw_url and domain

Code modified from MEIU22 package: https://github.com/osome-iu/MEIU22
"""
import os
import sys
import pandas as pd
import concurrent.futures
import queue
import argparse
from infopolluter import get_logger, clean_url

# Number of parallel workers to send http requests
WORKERS = 20

if __name__ == "__main__":
    log_dir = "logs"
    logger = get_logger(
        log_dir=log_dir,
        full_log_path=os.path.join(log_dir, f"url_expansion.log"),
        also_print=True,
    )

    # in_fpath = "data/processed/streaming_data--2021-01-06.parquet"
    # out_fpath = "data/urls/streaming_data--2021-01-06.parquet"
    in_fpath = sys.argv[1]
    out_fpath = sys.argv[2]

    # if not os.path.exists(out_fpath):
    # Merge the files for each day
    url_df = pd.read_parquet(in_fpath)
    if "raw_url" not in url_df.columns:
        raise ValueError(
            "Expect input file is a df of Twitter posts with a column named 'raw_url'"
        )

    # Start to expand the URLs
    focal_df = url_df[url_df.raw_url.notna()]

    if len(focal_df) > 0:
        unique_raw_urls = list(focal_df.raw_url.unique())
        logger.info(f"Expand {len(unique_raw_urls)} urls")
        q = queue.Queue()

        def expand_domain(short_url):
            expanded_url, domain = clean_url(short_url)
            q.put([short_url, expanded_url, domain])

        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
            executor.map(expand_domain, unique_raw_urls)

        expanded_url_df = pd.DataFrame(
            q.queue, columns=["raw_url", "expanded_url", "domain"]
        )
        df = url_df.merge(expanded_url_df, on="raw_url")

    else:
        logger.error(f"Empty input (No URLs to expand)")
        expanded_url_df = pd.DataFrame(columns=["raw_url"])
        df = url_df.merge(expanded_url_df, on="raw_url")

    logger.info(f"Writing to {out_fpath}")
    df.to_parquet(out_fpath, engine="pyarrow", index=None)

    logger.info(f"Script {__name__} done")

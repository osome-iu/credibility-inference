"""
Extract information from raw Twitter data (only keep the fields relevant to our analysis. More below)
user_id, retweeted_uid, retweeted_tweet_id, tweet_id, timestamp, text
NOTE: a quoted tweet is not an explicit sign of endorsement, because the message can be different.
The account that quotes the tweet can add additional domains, so we ignore quotes. 
Inputs:
    .json.gz file

Outputs:
    Data frame (.parquet) with the following fields:
    - post_id (str): id_str of the tweet
    - user_id (str): id_str of user
    - timestamp (int): UNIX timestamp of post (always represents the time the base post was sent)
    - tweet_type (str): options -> {original, retweet, quote}
    - retweeted_user_id (str, None): id_str of the retweeted poster. Only filled if the base tweet
        is a retweet, otherwise None
    - retweeted_post_id (str, None): id_str of the retweeted tweet. Only filled if the base tweet
        is a retweet, otherwise None
    - from_quoted (bool): whether or not the entity was taken from an embedded quoted_status object.
        True = entities taken from the quoted status object
        False = entities from taken from the base tweet object

    Entity specific fields in addition to those above for:
    - text (str): full tweet text
    - raw_url (str): raw url

Code modified from MEIU22 package: https://github.com/osome-iu/MEIU22/tree/main/code/package
"""

# TODO: take care of rt, quotes, text (there might be urls in the text)
import gzip
import json
import os
import pandas as pd
import sys
from copy import deepcopy

from infopolluter.data import Tweet
from infopolluter import get_logger

##TODO: Add user screen name for easy inspection?


def extract_entities(tweets_path):
    """
    Extract entities from raw Twitter data.
    Return four record structures for the major data entities: text, url, media, hashtags

    Parameters:
    -----------
    - tweets_path (str): full path to .json.gz file of tweet
    - entity (str): type of entity to extract from tweet: {text, urls, media, hashtags, all}

    Returns:
    -----------
    - text_data (list of dict): URLs
    - url_data (list of dict): text

    Note: List objects are dictionary records, where each dict will become a row in a dataframe and dict keys are columns.
    More here: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.from_records.html
    """

    # For output data
    data = []
    # Script management variables
    total = 0
    quotes = 0
    num_skipped_posts = 0

    try:
        with gzip.open(tweets_path, "rb") as f:
            for line in f:
                total += 1
                try:
                    tweet = Tweet(json.loads(line.decode("utf-8")))

                    if not tweet.is_valid():
                        # Skip invalid tweets
                        continue

                    # Data dict shared only for different base-level entities
                    base_info = {
                        "post_id": tweet.get_post_ID(),
                        "user_id": tweet.get_user_ID(),
                        "user_screen": tweet.get_user_screenname(),
                        "timestamp": tweet.get_timestamp(),
                        "tweet_type": "original",
                        "retweeted_post_id": None,
                        "retweeted_user_id": None,
                        "retweeted_user_screen": None,
                    }

                    # Skip a quoted status
                    if (not tweet.is_retweet) and tweet.is_quote:
                        quotes += 1
                        continue

                    if tweet.is_retweet:
                        base_info["tweet_type"] = "retweet"
                        base_info["retweeted_user_id"] = tweet.get_retweeted_user_ID()
                        base_info["retweeted_post_id"] = tweet.get_retweeted_post_ID()

                        tweet = tweet.retweet_object
                        base_info["retweeted_user_screen"] = tweet.get_user_screenname()
                        # Tweet can be Regular retweet or RT of a quoted status (both retweet and quote status objects are present)
                        # If RT of a quoted status, we COULD additionally extract URLs from the quoted_status object and label quoted.
                        # But quoting a tweet is not an endorsement, so we ignore it for now.
                        # if (tweet.is_retweet) and (tweet.is_quote):

                    ## Get text ##
                    base_info["text"] = tweet.get_text(clean=False)

                    ## Get urls ##
                    base_urls = tweet.get_URLs(recursive=False)

                    if len(base_urls) == 0:
                        base_info["raw_url"] = None
                        data.append(base_info)
                    else:
                        # b: long data for urls (so we can calculate user's credibility later)
                        # b: but we will group by (retweeter-retweeted later)
                        for url in base_urls:
                            url_record = deepcopy(base_info)
                            url_record["raw_url"] = url
                            data.append(url_record)

                except Exception as e:
                    logger.exception("Error parsing a tweet")
                    logger.info(e)
                    logger.error(tweet.post_object)
                    num_skipped_posts += 1
                    continue

        pct = round(((total - quotes) / total) * 100, 2)
        logger.info(
            f" - Extracted {total-quotes}/{total} ({pct}%) posts - Num. of weird skipped posts: {num_skipped_posts}"
        )
        return data

    except EOFError as e:
        logger.info(f" - Handling bad ending of a file...")
        return data


if __name__ == "__main__":
    log_dir = "logs"

    logger = get_logger(
        log_dir=log_dir,
        full_log_path=os.path.join(log_dir, f"extract_entities.log"),
        also_print=True,
    )
    # in_fpath = "data/raw/streaming_data--2021-01-06.json.gz"
    # out_fpath = "data/processed/streaming_data--2021-01-06.parquet"
    in_fpath = sys.argv[1]
    out_fpath = sys.argv[2]

    logger.info(f" - Extracting from: {os.path.basename(in_fpath)}...")
    data = extract_entities(in_fpath)

    logger.info(f" - Dumping to {out_fpath}...")
    entity_df = pd.DataFrame.from_records(data)
    entity_df.drop_duplicates(inplace=True)
    entity_df.to_parquet(out_fpath, index=None, engine="pyarrow")

    logger.info("Script complete")

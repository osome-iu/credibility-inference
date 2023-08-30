"""
Helper functions and global variables 
"""

import numpy as np
import inspect
import pandas as pd
import os
import logging
import sys
import tqdm

from cleantext import clean
from langdetect import detect

import preprocessor

## Using langdetect instead of pycld3 or fasttext because installation failed

LABEL_COL = "label"
SCORE_COL = "score"
user_col = "user_id"
DOMAIN_COL = "domain"
# user_col = "uid"
name_map = {"true_label": LABEL_COL, "mean_score_times": SCORE_COL, "uid": user_col}


DECIMALS = 4
THRESHOLD = 60  # if meanscore < THRESHOLD, label is 1 (user is bad)
FILL_NAN = -1


def threshold_label(x, default=FILL_NAN):
    # return binary label based on a threshold. If x is nan, return a default value
    if x == FILL_NAN:
        return default
    elif x < THRESHOLD:
        return 1
    else:
        return 0


def get_score(idxs, labels, fill_nan=FILL_NAN):
    """
    Get user scores based on the domains they've shared
    - labels: np.array of domain labels where labels[i] is the label of domain i
    - idxs: np.array of index of shared domains
    - fill_nan: replace nans with a value, default -1 if no label is present for any domain
    """
    domain_scores = labels[idxs]
    known = domain_scores[np.argwhere(~np.isnan(domain_scores))]
    score = np.mean(known)
    if np.isnan(score) and fill_nan is not None:
        score = fill_nan
    known_frac = len(known) / len(domain_scores)
    return score, known_frac


def get_account_labels(label_path, return_df=False):
    """
    Return dictionary {account_id: label}
    Option to return a pandas dataframe or dictionary.
    """
    labels = pd.read_csv(label_path)
    for col in name_map.keys():
        assert col in labels.columns

    labels = labels.rename(columns=name_map)
    labels = labels.astype({user_col: str})
    labels.dropna(inplace=True, subset=SCORE_COL)
    assert labels[SCORE_COL].isna().sum() == 0
    if return_df is True:
        return labels
    else:
        labels = labels[[user_col, "label"]].set_index(keys=user_col)
        label_dict = labels.to_dict("index")
        return label_dict


############################################################
# LANGUAGE UTILITY
############################################################
def clean_text(text, keep_lang="en"):
    """
    A convenience function for cleantext.clean because it has an ugly amount
    of parameters.
    keep_lang: only return cleaned text in this language
    """
    try:
        text = clean(
            text,
            fix_unicode=True,  # fix various unicode errors
            to_ascii=True,  # transliterate to closest ASCII representation
            lower=True,  # lowercase text
            no_line_breaks=True,  # fully strip line breaks as opposed to only normalizing them
            no_urls=True,  # replace all URLs with a special token
            no_emoji=True,  # remove emojis
            no_emails=True,  # replace all email addresses with a special token
            no_phone_numbers=True,  # replace all phone numbers with a special token
            no_numbers=False,  # replace all numbers with a special token
            no_digits=False,  # replace all digits with a special token
            no_currency_symbols=False,  # replace all currency symbols with a special token
            no_punct=True,  # remove punctuations
            replace_with_punct="",  # instead of removing punctuations you may replace them
            replace_with_url="",
            replace_with_email="",
            replace_with_phone_number="<PHONE>",
            replace_with_number="<NUMBER>",
            replace_with_digit="0",
            replace_with_currency_symbol="<CUR>",
            lang="en",  # set to 'de' for German special handling
        )

        lang = detect(text)
        if lang == keep_lang:
            return text
        else:
            return None
    except:
        return None


# def is_english(string):
#     if string is None or string == "":
#         return False
#     lang, proba, isReliable, _ = cld3.get_language(string)
#     english = lang == "en"
#     return isReliable and english


def preprocess_text(text, keep_lang="en"):
    try:
        # preprocess removes emojis, url, mentions, numbers. Can be reconfigured to include those.
        # Stopwords are not removed to retain readability when we want to manually check tweets.
        text = preprocessor.clean(text)
        special_char = '.@_!$%^&*()<>?/\|}{~:;,[]"'
        tokens = text.split(" ")
        numbered_tokens = [t for t in tokens if not has_numbers(t)]
        hashtags = [t.replace("#", "") for t in tokens if t.startswith("#")]
        raw_text = " ".join([t for t in tokens if not has_numbers(t)])
        text = raw_text.lower().translate(
            {ord(ch): " " for ch in "0123456789" + special_char}
        )
        # removes spaces in-between words
        text = " ".join(text.split())
        lang = detect(text)
        if lang == keep_lang:
            return text
        else:
            return None
    except:
        return None


def has_numbers(string):
    return any(char.isdigit() for char in string)


def warn(text: str):
    """
    Pre-pends a red-colored 'WARNING: ' to [text]. This is a printed warning and cannot be suppressed.
    :param text: Warning message
    :return: 'WARNING: [text]'
    """
    print("\033[91m" + "WARNING: " + "\033[0m" + text)


def get_dict_val(dictionary: dict, key_list: list = [], default=None):
    """
    Return `dictionary` value at the end of the key path provided
    in `key_list`.
    Indicate what value to return based on the key_list provided.
    For example, from left to right, each string in the key_list
    indicates another nested level further down in the dictionary.
    If no value is present, return `default` (default=None).
    Parameters:
    ----------
    - dictionary (dict) : the dictionary object to traverse
    - key_list (list) : list of strings indicating what dict_obj
        item to retrieve
    Returns:
    ----------
    - key value (if present) or None (if not present)
    Raises:
    ----------
    - TypeError
    Examples:
    ---------
    # Create dictionary
    dictionary = {
        "a" : 1,
        "b" : {
            "c" : 2,
            "d" : 5
        },
        "e" : {
            "f" : 4,
            "g" : 3
        },
        "h" : 3
    }
    ### 1. Finding an existing value
    # Create key_list
    key_list = ['b', 'c']
    # Execute function
    get_dict_val(dictionary, key_list)
    # Returns
    2
    ~~~
    ### 2. When input key_path doesn't exist
    # Create key_list
    key_list = ['b', 'k']
    # Execute function
    value = get_dict_val(dictionary, key_list)
    # Returns NoneType because the provided path doesn't exist
    type(value)
    NoneType
    """
    if not isinstance(dictionary, dict):
        raise TypeError("`dictionary` must be of type `dict`")

    if not isinstance(key_list, list):
        raise TypeError("`key_list` must be of type `list`")

    retval = dictionary
    for k in key_list:
        # If retval is not a dictionary, we're going too deep
        if not isinstance(retval, dict):
            return default

        if k in retval:
            retval = retval[k]

        else:
            return default
    return retval


def normalize(v):
    norm = np.linalg.norm(v, ord=1)
    if norm == 0:
        return v
    return v / norm


def remove_illegal_kwargs(adict, amethod):
    # remove a keyword from a dict if it is not in the signature of a method
    new_dict = {}
    argspec = inspect.getargspec(amethod)
    legal = argspec.args
    for k, v in adict.items():
        if k in legal:
            new_dict[k] = v
    return new_dict


# Loggers
## Logging Handler for tqdm
class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


############################################################
def get_logger(log_dir, full_log_path, also_print=False, tqdm=False):
    """Create logger."""

    # Create log_dir if it doesn't exist already
    try:
        os.makedirs(f"{log_dir}")
    except:
        pass

    # Create logger and set level
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    # Configure file handler
    formatter = logging.Formatter(
        fmt="%(asctime)s-%(name)s-%(levelname)s-%(message)s",
        datefmt="%Y-%m-%d_%H:%M:%S",
    )
    fh = logging.FileHandler(f"{full_log_path}")
    fh.setFormatter(formatter)
    fh.setLevel(level=logging.INFO)
    logger.addHandler(fh)

    if tqdm:
        logger.addHandler(TqdmLoggingHandler())
    # If also_print is true, the logger will also print the output to the
    # console in addition to sending it to the log file
    if also_print:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        ch.setLevel(level=logging.INFO)
        logger.addHandler(ch)

    return logger

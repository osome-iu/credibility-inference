from gensim.models.keyedvectors import KeyedVectors
import numpy as np
from credinference.util import get_dict_val, user_col
from tqdm import tqdm


def load_glove_embedding(file_glove_wiki: str):
    """
    Read the pretrained GloVe model: https://groups.google.com/g/gensim/c/G40gps4ngPw?pli=1
    file_glove_wiki e.g: glove.twitter.27B.200d.txt
    Return glovewiki (gensim.models.keyedvectors.KeyedVectors)
    This is a dict mapping words - embedding vectors, e.g: access using glove['world']
    """
    # if the load time is too long, save a new file with save_word2vec_format()
    out_file = file_glove_wiki.split(".txt")[0] + ".w2vformat.txt"
    glovewiki = KeyedVectors.load_word2vec_format(
        file_glove_wiki, binary=False, no_header=True
    )
    # KeyedVectors.save_word2vec_format()
    return glovewiki


def sentence2vector(text, w2v_wiki, emb_type="glove"):
    # TODO: we can use different types of string aggregation here
    """
    Get vector representation for a sentence (mean pooling)
    text: str
    resulting vec has the same shape as pretrained embeddings: seq_emb: np array: shape: (,n_dimension)
    w2v_wiki: pretrained GloVe (KeyedVectors object) or Fasttext model (FastText object): dict: {word: vec}. Vec shape: (,n_dimension)
    """

    if emb_type == "glove":
        emb_dimension = w2v_wiki.vector_size
    elif emb_type == "fasttext":
        emb_dimension = w2v_wiki.get_dimension()
    else:
        raise TypeError("The specified embedding type is incorrect!")
    tokens_emb = [
        w2v_wiki[token] if token in w2v_wiki else np.zeros(emb_dimension)
        for token in text.split(" ")
    ]
    seq_mat = np.vstack(tokens_emb)
    seq_emb = np.mean(seq_mat, axis=0)
    return seq_emb


def get_text2vec_mapping(
    df, w2v_wiki, emb_type: str = "glove", text_col: str = "cleaned_text"
):
    """
    Make a new column by mapping a user's cleaned text into an embedding vector
    df: df with at least 2 columns: that stands for user id and associated posts
    Return dict of user: words
    """

    df = df.loc[:, [user_col, text_col]]
    df = df.groupby([user_col], as_index=False).agg({text_col: " ".join})
    # resulting df has 2 cols: uid and text_col
    df = df.set_index(user_col)
    user2words_dict = df.to_dict(orient="index")
    user_vectors = dict()
    for user in tqdm(
        user2words_dict.keys(), desc="Mapping users to text embedding vectors"
    ):
        sentence = get_dict_val(user2words_dict, [user, text_col])
        user_vectors[user] = sentence2vector(sentence, w2v_wiki, emb_type=emb_type)
    return user_vectors

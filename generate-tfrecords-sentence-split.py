import tensorflow as tf
import tensorflow_text as text
import tensorflow_hub as hub

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sentence_splitter import split_text_into_sentences
from typing import List, Callable, Tuple
import pandas as pd
import numpy as np
import random
import tqdm

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


train_df = pd.read_csv(
    "./data/train_data.txt",
    engine="python",
    sep=" ::: ",
    names=["id", "movie", "genre", "summary"],
)

test_df = pd.read_csv(
    "./data/test_data_solution.txt",
    engine="python",
    sep=" ::: ",
    names=["id", "movie", "genre", "summary"],
)


# Split the data using train_test_split from sklearn
train_shuffled = train_df.sample(frac=1)
train_df_new, val_df = train_test_split(train_shuffled, test_size=0.1)

print(f"Number of training samples: {len(train_df_new)}.")
print(f"Number of validation samples: {len(val_df)}.")
print(f"Number of test examples: {len(test_df)}.")


le = LabelEncoder()
le.fit(train_df_new["genre"].values) 

train_df_new["genre"] = le.transform(train_df_new["genre"].values)
val_df["genre"] = le.transform(val_df["genre"].values)
test_df["genre"] = le.transform(test_df["genre"].values)


def set_tokenizer(preprocessor_path: str) -> Callable:
    """ Decorator to set the desired tokenizer for a tokenizing
        function from a TensorFlow Hub URL.
        
    Arguments:
        preprocessor_path {str} -- URL of the TF-Hub preprocessor.
    
    Returns:
        Callable -- A function with the `tokenizer` attribute set.
    """

    def decoration(func: Callable):
        # Loading the preprocessor from TF-Hub
        preprocessor = hub.load(preprocessor_path)

        # Setting an attribute called `tokenizer` to
        # the passed function
        func.tokenizer = preprocessor.tokenize
        return func

    return decoration


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    elif not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


@set_tokenizer(
    preprocessor_path="https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
)
def _tokenize_text(text: List[str]) -> Tuple[tf.RaggedTensor, List[int]]:
    """Tokenizes a list of sentences.
        Args:
            text (List[str]): A list of sentences.
        Returns:
            Tuple[tf.RaggedTensor, List[int]]: Tokenized and indexed sentences, list containing
            the number of tokens per sentence.
        """
    token_list = _tokenize_text.tokenizer(text)
    token_lens = [tokens.flat_values.shape[-1] for tokens in token_list]
    return token_list, token_lens


def serialize_composite(rt):
    """Serializes as a Ragged feature."""
    components = tf.nest.flatten(rt, expand_composites=True)
    return tf.io.serialize_tensor(
        tf.stack([tf.io.serialize_tensor(t) for t in components])
    ).numpy()


def get_serialized_text_features(features):
    """Serializes all the Ragged features."""
    tokens = features["tokens"]
    tokens = serialize_composite(tokens)
    tokens = _bytes_feature(tokens)

    lens = features["lens"]
    lens = tf.ragged.constant([lens])
    lens = serialize_composite(lens)
    lens = _bytes_feature(lens)

    sentence_idx = list(range(features["num_sentences"]))
    sentence_idx = tf.ragged.constant([sentence_idx])
    sentence_idx = serialize_composite(sentence_idx)
    sentence_idx = _bytes_feature(sentence_idx)

    return tokens, lens, sentence_idx


def create_example(row):
    """Creates one TFRecord example."""
    summary = row["summary"]
    label = row["genre"]

    description = bytes(summary, encoding="utf-8")
    description_tokens, description_len = _tokenize_text(
        split_text_into_sentences(summary, language="en")
    )

    features = {
        "tokens": description_tokens,
        "lens": description_len,
        "num_sentences": len(description_len),
    }
    (text_tokens, text_lens, text_sentence_idx) = get_serialized_text_features(features)

    feature = {
        "summary": _bytes_feature(description),
        "summary_tokens": text_tokens,
        "summary_sentence_indices": text_sentence_idx,
        "summary_num_sentences": _int64_feature(description_len),
        "summary_tokens_len": text_lens,
        "label": _int64_feature(label),
    }
    feature = tf.train.Features(feature=feature)
    example = tf.train.Example(features=feature)
    return example


def write_tfrecords(file_name, data):
    """Serializes the data as string."""
    with tf.io.TFRecordWriter(file_name) as writer:
        for i, row in data.iterrows():
            example = create_example(row)
            writer.write(example.SerializeToString())


TFRECORDS_DIR = "gs://variable-length-sequences-tf/tfrecords-sentence-splitter"
CHUNK_SIZE = 100



def write_data(data, chunk_size, files_prefix):
    """Serializes data as TFRecord shards."""
    example_counter = 0
    chunk_count = 1
    for i in tqdm.tqdm(range(0, data.shape[0], chunk_size)):
        chunk = data.iloc[i : i + chunk_size, :]
        file_name = f"{TFRECORDS_DIR}/{files_prefix}-{chunk_count:02d}.tfrecord"
        write_tfrecords(file_name, chunk)
        example_counter += chunk.shape[0]
        chunk_count += 1
    return example_counter


train_example_count = write_data(train_df_new, CHUNK_SIZE, "train")
# val_example_count = write_data(val_df, CHUNK_SIZE, "val")
# test_example_count = write_data(test_df, CHUNK_SIZE, "test")
# print(train_example_count, val_example_count, test_example_count)

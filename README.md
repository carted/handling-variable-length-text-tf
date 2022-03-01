# Handling variable-length text sequences in TensorFlow

This repository code for the following series of blog posts:

* Handling variable-length text sequences in TensorFlow (Part I)
* Handling variable-length text sequences in TensorFlow (Part II)
* Handling variable-length text sequences in TensorFlow (Part III)

We provide code in the form of Jupyter Notebook so that you can experiment with the methods interactively. 

## What is it about?

Text data comes in different shapes and forms. Sometimes they come as sequences of characters, sometimes as sequences
of words, and sometimes even as sequences of sentences, and so on. Now, for machine learning (ML) algorithms to process
text sequences in batches the batches need to have uniform-length sequences. However, text sequences can come in varying
lengths. 

In this project, we implement different strategies to handle variable-length sequences in TensorFlow with a focus
on performance. We will discuss the pros and cons of each strategy along with their implementations in TensorFlow.
We have successfully applied some of these strategies to the large-scale data here at Carted and have greatly benefited
from them. We hope you’ll be able to apply them in your own projects as well.

## General setup

We’ll be using [this dataset](https://www.kaggle.com/hijest/genre-classification-dataset-imdb) from Kaggle which
concerns a text classification problem. Specifically, given some description of a movie, we need to predict its
genre. Each description consists of multiple sentences and there are 27 unique genres (such as action, adult, adventure,
animation, biography, comedy, etc.). 

As a disclaimer, our focus is on designing efficient data pipelines for handling variable-length text sequences
that can help us reduce compute waste. But we will also see how to use these pipelines to train text classification
models for completeness.

**Note**: One central theme around our code is to be able to process text sequences in the following manner. Padding a
batch of sequences with respect to the maximum sequence length of the batch instead of a fixed sequence length. 

## Navigating through the notebooks

* `smart-batching-shallow-mlp.ipynb`: Shows how to train a text classifier using simple models consisting of
  embeddings, GRUs, and fully-connected layers. 
* `bert/`
  * `data-preparation.ipynb`: Shows how to prepare text data into TensorFlow Records (TFRecords) with tokenization. The
    corresponding modeling notebook is present at `bert/train-vanilla-bert.ipynb`.
  * `data-preparation-sentence-splitter.ipynb`: Treats each movie description as a sequence of sentences and 
    serializes them into TFRecords. The corresponding modeling notebook is present at `bert/train-model-split-sentence.ipynb`.

All the results discussed in the above-mentioned articles can be found at the following links:

* https://wandb.ai/carted/smart-batching-simpler-models (simpler models)
* https://wandb.ai/carted/batching-experiments (BERT models)

## Questions

Feel free to open an issue and let us know.


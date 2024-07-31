from typing import List

import numpy as np
from gensim.models.word2vec import Word2Vec


def vectorizer(
    corpus: List[List[str]], model: Word2Vec, num_features: int = 100
) -> np.ndarray:
    """
    This function takes a list of tokenized text documents (corpus) and a pre-trained
    Word2Vec model as input, and returns a matrix where each row represents the
    vectorized form of a document.

    Args:
        corpus : list
            A list of text documents that needs to be vectorized.

        model : Word2Vec
            A pre-trained Word2Vec model that will be used to vectorize the corpus.

        num_features : int
            The size of the vector representation of each word. Default is 100.

    Returns:
        corpus_vectors : numpy.ndarray
            A 2D numpy array where each row represents the vectorized form of a
            document in the corpus.
    """
    # TODO
    def data_stream(corpus):
        for doc in corpus:
            yield doc
    
    def document_vectorizer(doc: List[str], model: Word2Vec):
        feature_vector =  np.zeros((num_features,), dtype="float32")
        nwords = 0

        for token in doc:
            if token in model.wv.key_to_index:
                nwords += 1
                feature_vector = np.add(feature_vector, model.wv.get_vector(token))
        if nwords > 1:
            feacture_vector = np.divide(feature_vector, nwords)
        return feacture_vector
    
    corpus_vector = np.zeros((len(corpus), num_features), dtype="float32")
    for i, doc in enumerate(data_stream(corpus)):
        corpus_vector[i]=document_vectorizer(doc, model)
    return corpus_vector

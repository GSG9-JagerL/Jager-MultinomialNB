#  Copyright (c) 2021 Jäger. All rights reserved.
import string

import numpy as np
from typing import List


class CountVectorizer:

    def __init__(self, stop_words: List[str], max_features: int = None):
        self.vocabularies = []
        self.stop_words = stop_words
        self.max_features = max_features
        pass

    def fit(self, raw_documents) -> None:
        raw_documents_all_in_1 = ' '.join(raw_documents)
        all_words = np.array(self.text_process(raw_documents_all_in_1), dtype=object)
        unique, counts = np.unique(all_words, return_counts=True)
        bag_of_words = np.transpose(np.array([unique, counts], dtype=object))
        self.vocabularies = bag_of_words[bag_of_words[:, 1].argsort()]  # sort
        self.vocabularies = self.vocabularies[::-1]  # descending order
        if self.max_features is not None:
            self.vocabularies = self.vocabularies[:self.max_features]

    def transform(self, raw_documents, tf_idf=False):
        vectorized_document = np.empty(shape=(len(raw_documents), len(self.vocabularies)), dtype=object)
        for i in range(len(raw_documents)):
            words = self.text_process(raw_documents[i])
            for j in range(len(self.vocabularies)):
                vectorized_document[i, j] = words.count(self.vocabularies[:, 0][j])
        if tf_idf:
            return vectorized_document, CountVectorizer.idf(vectorized_document)
        else:
            return vectorized_document

    def text_process(self, abstract: str) -> List[str]:
        return [word.lower() for word in ''.join([char for char in abstract if char not in string.punctuation]).split()
                if
                word not in self.stop_words]

    # noinspection PyPep8Naming
    @staticmethod
    def idf(vectorized_document: np.ndarray) -> np.ndarray:
        N = len(vectorized_document)
        D = [len(vectorized_document[vectorized_document[:, j] != 0]) for j in range(vectorized_document.shape[1])]
        D = np.array(D, dtype=np.float)
        IDF = np.log(N+1 / D+1)+1
        return IDF

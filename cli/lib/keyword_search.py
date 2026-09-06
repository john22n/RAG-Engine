import pickle
import os
import math
from nltk.stem import PorterStemmer
from collections import defaultdict, Counter
from .search_utils import (
    CACHE_DIR,
    CACHE_INDEX_PATH,
    CACHE_DOCMAP_PATH,
    CACHE_TERM_FREQ_PATH,
    Movie,
    load_movies
)

from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    load_movies,
    load_stopwords,
    preprocessed_text,
)

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict] | None:
    IC = InvertedIndex()
    IC.load()
    query_tokens = tokenize_text(query)
    seen, results = set(), []
    for query_token in query_tokens:
        matching_doc_ids = IC.get_document(query_token)
        for doc_id in matching_doc_ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            doc = IC.docmap[doc_id]
            results.append(doc)
            if len(results) >= limit:
                return results
    return results


def tokenize_text(text: str) -> list[str]:
    stemmer = PorterStemmer()
    text = preprocessed_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        token = stemmer.stem(token)
        if token:
            valid_tokens.append(token)
    return valid_tokens

def filter_tokens(tokens: list[str]) -> list[str]:
    filters = load_stopwords()
    res = []
    for token in tokens:
        if token not in filters:
            res.append(token)
    return res

def tokenize_text_helper(text: str) -> str:
    token = tokenize_text(text)
    if len(token) != 1:
        raise Exception("length not 1")
    return token[0]


class InvertedIndex:

    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, Movie] = {}
        self.term_frequencies = defaultdict(Counter)

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = filter_tokens(tokenize_text(text))
        for token in tokens:
            self.term_frequencies[doc_id][token] += 1

        for token in set(tokens):
            self.index[token].add(doc_id)

    def get_document(self, term: str) -> list[int]:
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def build(self) -> None:
        movies = load_movies()
        for movie in movies:
            doc_id = movie["id"]
            doc_description = f"{movie['title']} {movie['description']}"
            self.docmap[doc_id] = movie
            self.__add_document(doc_id, doc_description)

    def get_tf(self, doc_id: int, term: str) -> int:
        return self.term_frequencies.get(doc_id, {}).get(term, 0)

    def get_idf(self, text: str) -> float:
        docs = self.docmap
        num_of_docs = len(self.docmap.keys())
        frequency = self.get_doc_frequency(text)
        return math.log((num_of_docs + 1) / (frequency + 1))

    def get_tf_idf(self, doc_id: int, term: str) -> float:
        return self.get_tf(doc_id, term) * self.get_idf(term)

    def get_doc_frequency(self, term: str) -> int:
        frequency = 0
        for doc in self.docmap:
            if term in self.term_frequencies[doc]:
                frequency += 1
        return frequency

    def get_bm25_idf(self, term: str) -> float:
        num_of_docs = len(self.docmap.keys())
        frequency = self.get_doc_frequency(term)
        return math.log((num_of_docs - frequency + 0.5) / (frequency + 0.5) + 1)

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(CACHE_INDEX_PATH, "wb") as index_file:
            pickle.dump(self.index, index_file)

        with open(CACHE_DOCMAP_PATH, "wb") as docmap_file:
            pickle.dump(self.docmap, docmap_file)

        with open(CACHE_TERM_FREQ_PATH, "wb") as term_freq_file:
            pickle.dump(self.term_frequencies, term_freq_file)

    def load(self) -> tuple[dict, dict] | None:
        try:
            with open(CACHE_INDEX_PATH, "rb") as index_file:
                self.index = pickle.load(index_file)

            with open(CACHE_DOCMAP_PATH, "rb") as doc_file:
                self.docmap = pickle.load(doc_file)

            with open(CACHE_TERM_FREQ_PATH, "rb") as term_freq_file:
                self.term_frequencies = pickle.load(term_freq_file)

        except FileNotFoundError:
            print(f"Error: file not found")


def build_command() -> None:
    Index = InvertedIndex()
    Index.build()
    Index.save()

def tf_command(doc_id: int, term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf(doc_id, tokenize_text_helper(term))

def idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_idf(tokenize_text_helper(term))

def tf_idf_command(doc_id: int, term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    tokenized = tokenize_text_helper(term)
    return  idx.get_tf_idf(doc_id, tokenized)

def bm25_idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    tokenized_term = tokenize_text_helper(term)
    return idx.get_bm25_idf(tokenized_term)



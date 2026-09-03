from nltk.stem import PorterStemmer
from .search_utils import CACHE_DIR, load_movies, CACHE_INDEX_PATH, CACHE_DOCMAP_PATH, Movie
from collections import defaultdict
import pickle
import os

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

class InvertedIndex:

    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, Movie] = {}

    def __add_document(self, doc_id: int, text: str, map: dict) -> None:
        tokens = filter_tokens(tokenize_text(text))
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
            self.__add_document(doc_id, doc_description, self.index)

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(CACHE_INDEX_PATH, "wb") as index_file:
            pickle.dump(self.index, index_file)
        with open(CACHE_DOCMAP_PATH, "wb") as docmap_file:
            pickle.dump(self.docmap, docmap_file)

    def load(self) -> tuple[dict, dict] | None:
        try:
            with open(CACHE_INDEX_PATH, "rb") as index_file:
                self.index = pickle.load(index_file)

            with open(CACHE_DOCMAP_PATH, "rb") as doc_file:
                self.docmap = pickle.load(doc_file)

        except FileNotFoundError:
            print(f"Error: file not found")

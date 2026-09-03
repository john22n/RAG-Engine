from .search_utils import CACHE_DIR, load_movies, CACHE_INDEX_PATH, CACHE_DOCMAP_PATH, Movie
from .keyword_search import tokenize_text
from collections import defaultdict
import pickle
import os

class InvertedIndex:

    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, Movie] = {}

    def __add_document(self, doc_id: int, text: str, map: dict) -> None:
        tokens = tokenize_text(text)
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
                index= pickle.load(index_file)

            with open(CACHE_DOCMAP_PATH, "rb") as doc_file:
                docmap= pickle.load(doc_file)

            return index, docmap

        except FileNotFoundError:
            print(f"Error: file not found")

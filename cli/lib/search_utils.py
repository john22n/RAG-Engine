import json
from nltk.stem import PorterStemmer
import os
import string
from typing import Any, TypedDict


class Movie(TypedDict):
    id: int
    title: str
    description: str

class SearchResult(TypedDict):
    id: int
    title: str
    document: str
    score: float
    metadata: dict[str, Any]

class GoldenTestCase(TypedDict):
    query: str
    relavent_docs: list[str]

class GoldenDataset(TypedDict):
    test_cases: list[GoldenTestCase]


DEFAULT_ALPHA = 0.5
RRF_K = 60
SEARCH_MULTIPLIER = 5
DOCUMENT_PREVIEW_LENGTH = 100
DEFAULT_SEARCH_LIMIT = 5
SCORE_PRECISION = 3

SCORE_PERCISION = 3
BM25_K1 = 1.5
BM25_B = 0.75

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
GOLDEN_DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "golden_dataset_json")

CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
CACHE_INDEX_PATH = os.path.join(PROJECT_ROOT, "cache", "index.pkl")
CACHE_DOCMAP_PATH = os.path.join(PROJECT_ROOT, "cache", "docmap.pkl")
CACHE_TERM_FREQ_PATH = os.path.join(PROJECT_ROOT, "cache", "term_frequencies.pkl")
CACHE_DOCS_LENGTH_PATH = os.path.join(PROJECT_ROOT, "cache", "docs_lengths.pkl")

DEFAULT_CHUNK_SIZE = 200
DEFAULT_CHUNK_OVERLAP = 1
DEFAULT_SEMANTIC_CHUNCH_SIZE = 4

MOVIE_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "movie_embeddings.npy")
CHUNK_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
CHUNK_METADATA_PATH = os.path.join(CACHE_DIR, "chunk_metadata.json")

def preprocessed_text(text: str) -> str:
    text = text.lower()
    return text.translate(str.maketrans("", "", string.punctuation))

def load_movies() -> list[Movie]:
    with open(DATA_PATH, 'r') as f:
        data = json.load(f)
    return data["movies"]

def format_search_result(
        doc_id: int, title: str, document: str, score: float, **metadata: Any
        ) -> SearchResult:
    """ create standardized search result

    Args:
        doc_id: Document ID
        title: Document title
        document: Display text (usually short description)
        score: Relevance/similarity score
        **metadata: Additional metadata to include

    Returns:
        Dictionary representation of search result:
    """
    return {
            "id": doc_id,
            "title": title,
            "document": document,
            "score": round(score, SCORE_PRECISION),
            "metadata": metadata if metadata else {}
        }

def load_golden_dataset() -> GoldenDataset:
    with open(GOLDEN_DATASET_PATH, "r") as f:
        return json.load(f)

def load_stopwords() -> list[str]:
    stemmer = PorterStemmer()
    with open(STOPWORDS_PATH, 'r') as f:
        return [stemmer.stem(preprocessed_text(word)) for word in f.read().splitlines()]

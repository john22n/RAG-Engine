from collections import defaultdict
import os
from typing import NotRequired, TypedDict

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch, SemanticSearchResult
from .search_utils import Movie, CACHE_INDEX_PATH, load_movies

class WeightedSearchResults(TypedDict):
    keyword: float
    semantic: float
    title: str
    description: str
    hybrid: NotRequired[float]

class HybridSearch:
    def __init__(self, documents: list[Movie]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)
        self.idx = InvertedIndex()

        if not os.path.exists(CACHE_INDEX_PATH):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> tuple[list, dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[WeightedSearchResults]:
        sorted_keyword_res, documents = self._bm25_search(query, limit * 500)
        semantic_results: list[dict] = self.semantic_search.search_chunks(query, limit * 500)

        keyword_normalized = normalize([item[1] for item in sorted_keyword_res])
        keyword_paired = [(id, norm) for (id, __), norm in zip(sorted_keyword_res, keyword_normalized)]

        semantic_normalized = normalize([sm['score'] for sm in semantic_results])
        semantic_paired = [(obj, norm) for obj, norm in zip(semantic_results, semantic_normalized)]

        document_scores: dict[int, WeightedSearchResults] = {}

        for doc_id, score in keyword_paired:
            title = documents[doc_id]['title']
            description = documents[doc_id]['description']
            entry = document_scores.setdefault(doc_id, {"keyword": 0.0, "semantic": 0.0, "hybrid": 0.0, "title": "", "description": ""})
            entry["keyword"] = score
            entry["title"] = title
            entry["description"] = description

        for doc, score in semantic_paired:
            title = doc['title']
            description = doc['document']
            entry = document_scores.setdefault(doc['id'], {"keyword": 0.0, "semantic": 0.0, "hybrid": 0.0, "title": title, "description": description})
            entry["semantic"] = max(entry['semantic'], score)

        for doc in document_scores.values():
            hybrid = hybrid_score(doc['keyword'], doc['semantic'], alpha)
            doc['hybrid'] = hybrid

        sorted_scores: list[WeightedSearchResults] = sorted(
                document_scores.values(),
                key=lambda item: item.get('hybrid', 0),
                reverse=True
                )

        return sorted_scores[:limit]

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        raise NotImplementedError("RRF hybrid search is not implemented yet.")


def normalize(numbers: list[float]) -> list[float]:
    if not numbers:
        return []

    min_score = min(numbers)
    max_score = max(numbers)

    if max_score == min_score:
        return [1.0] * len(numbers)

    normalized_scores = []
    for n in numbers:
        normalized_scores.append((n - min_score) / (max_score - min_score))
    return normalized_scores

def search(query, alpha, limit) -> None:
    movies = load_movies()
    hybrid_search = HybridSearch(movies)
    res = hybrid_search.weighted_search(query, alpha, limit)
    for i, r in enumerate(res, start=1):
        print(f"{i}. {r['title']}")
        print(f"Hybrid Score: {r.get('hybrid', 0):.4f}")
        print(f"BM25: {r['keyword']:.4f}, Semantic: {r['semantic']:.4f}")
        print(f"{r['description'][:100]}...")

def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score


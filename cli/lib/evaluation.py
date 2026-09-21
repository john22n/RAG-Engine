from typing import TypedDict

from .hybrid_search import HybridSearch
from .search_utils import (
        load_golden_dataset,
        load_movies,
        )
from .semantic_search import SemanticSearch

def precision_at_k(
        retrieved_docs: list[str],
        relevant_docs: set[str],
        k: int = 5
    ) -> float:
    top_k =retrieved_docs[:k]
    relevant_count = 0
    for doc in top_k:
        if doc in relevant_docs:
            relevant_count += 1
    return relevant_count / k

def recall_at_k(
        search_retrieved: list[str],
        total_relevant: list[str],
        k: int = 5
        ) -> float:
    top_k = search_retrieved[:k]

    relevant_count = 0
    for doc in top_k:
        if doc in total_relevant:
            relevant_count += 1

    if relevant_count == 0:
        return 0.0

    return  relevant_count / len(total_relevant)

def f1_score(precision: float, recall: float) -> float:
    if (precision + recall) == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def evaluate_command(limit: int = 5) -> dict:
    movies = load_movies()
    golden_dataset  = load_golden_dataset()
    test_cases = golden_dataset['test_cases']

    semantic_search = SemanticSearch()
    semantic_search.load_or_create_embeddings(movies)
    hybrid_search = HybridSearch(movies)

    total_precision = 0
    results_by_query = {}

    for test_case in test_cases:
        query = test_case['query']
        relevant_docs = set(test_case['relevant_docs'])
        search_results = hybrid_search.rrf_search(query, k=60, limit=limit)
        retrieved_docs = []
        for id, res in search_results:
            title = res.get('title', '')
            if title:
                retrieved_docs.append(title)

        precision = precision_at_k(retrieved_docs, relevant_docs, limit)
        recall = recall_at_k(retrieved_docs, relevant_docs, limit)
        fi = f1_score(precision, recall)

        results_by_query[query] = {
                'recall': recall,
                'precision': precision,
                'f1_score': fi,
                'retrieved': retrieved_docs[:limit],
                'relevant': list(relevant_docs)
                }
        total_precision += precision

    return {
            'test_cases_count': len(test_cases),
            'limit': limit,
            'results': results_by_query
            }



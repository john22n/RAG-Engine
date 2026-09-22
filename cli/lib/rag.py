from typing import TypedDict

from .llm_utils import LLM

from .hybrid_search import HybridSearch
from .search_utils import load_movies

def rag_command(query: str, limit: int = 5, command: str = "rag") -> dict:
    llm = LLM()
    movies = load_movies()
    hybrid_search = HybridSearch(movies)

    search_results = hybrid_search.rrf_search(query, 60, limit)

    titles = []
    descriptions = []

    for id, res in search_results:
        title = res.get('title', '')
        description = res.get('description', '')
        if title and description:
            title_description = title + ": " + description
            titles.append(title)
            descriptions.append(title_description)

    docs_str = ", ".join(descriptions)

    rag_res = llm.rag_results(query, docs_str)

    summarize_res = llm.summarize_results(query, docs_str)

    citation_res = llm.citation_results(query, docs_str)

    question_res = llm.question_results(query, docs_str)

    return {
            "search_results": titles,
            "rag_results": rag_res,
            "summary": summarize_res,
            "citation": citation_res,
            "question": question_res
            }










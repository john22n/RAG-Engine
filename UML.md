# RAG Search Engine Architecture

This repository is a movie-focused Retrieval-Augmented Generation (RAG) search engine. It supports keyword, semantic, hybrid, image-based search, LLM-generated answers, and retrieval evaluation.

```text
                                   ┌──────────────────────────┐
                                   │          User            │
                                   │ text query / image / CLI │
                                   └────────────┬─────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         CLI Interface Layer                         │
│                                                                     │
│ Keyword │ Semantic │ Hybrid │ RAG │ Image Search │ Image Q&A │ Eval │
└────┬─────────┬──────────┬──────┬─────────┬─────────────┬───────┬────┘
     │         │          │      │         │             │       │
     ▼         ▼          └──┬───┘         ▼             │       │
┌──────────┐ ┌──────────────────┐   ┌──────────────────┐  │       │
│Inverted  │ │   HybridSearch   │   │MultimodalSearch  │  │       │
│Index     │ │                  │   │                  │  │       │
│          │ │ ┌──────────────┐ │   │ CLIP image/text  │  │       │
│BM25 /    │◀┤ │ Keyword BM25 │ │   │ embeddings       │  │       │
│TF-IDF    │ │ └──────────────┘ │   └────────┬─────────┘  │       │
└────┬─────┘ │                  │            │            │       │
     │       │ ┌──────────────┐ │            │            │       │
     │       │ │  Chunked     │ │            │            │       │
     │       │ │  Semantic    │ │            │            │       │
     │       │ │  Search      │ │            │            │       │
     │       │ └──────┬───────┘ │            │            │       │
     │       │        │         │            │            │       │
     │       │   ┌────▼─────┐   │            │            │       │
     │       │   │ Weighted │   │            │            │       │
     │       │   │ fusion / │   │            │            │       │
     │       │   │   RRF    │   │            │            │       │
     │       │   └────┬─────┘   │            │            │       │
     │       └────────┼─────────┘            │            │       │
     │                ┃                      │            │       │
     │                ▼                      │            │       │
     │       ┌─────────────────┐             │            │       │
     │       │ Ranked movies   │◀────────────┘            │       │
     │       └────────┬────────┘                          │       │
     │                │                                   │       │
     │                ├──────────────────┐                │       │
     │                ▼                  ▼                ▼       │
     │       ┌─────────────────┐  ┌───────────────────────────┐   │
     │       │ Cross-encoder   │  │            LLM            │◀──┘
     │       │ or LLM reranker │  │                           │
     │       └────────┬────────┘  │ query enhancement         │
     │                │           │ RAG answers / summaries   │
     │                │           │ citations / image Q&A     │
     │                │           └─────────────┬─────────────┘
     │                └──────────────┬──────────┘
     │                               ▼
     │                     ┌───────────────────┐
     │                     │ CLI search result │
     │                     │ or generated      │
     │                     │ answer            │
     │                     └───────────────────┘
     │
     ▼
┌────────────────────────── Storage ──────────────────────────────────┐
│ data/movies.json         Movie titles and descriptions             │
│ data/stopwords.txt       Keyword-search filtering                  │
│ data/golden_dataset.json Expected results for evaluation           │
│ cache/*.pkl              Inverted index and term statistics        │
│ cache/*.npy / *.json     Movie/chunk embeddings and metadata       │
└────────────────────────────────────────────────────────────────────┘

External services and models:
  • Sentence Transformers → semantic embeddings
  • CLIP                  → image-to-movie similarity
  • CrossEncoder          → result reranking
  • OpenRouter/OpenAI API → query enhancement and generated answers
```

The primary RAG path is:

**Query → BM25 and semantic search → reciprocal-rank fusion → relevant movie descriptions → LLM prompt → generated answer.**

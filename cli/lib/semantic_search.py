from sentence_transformers import SentenceTransformer
from typing import Any, TypedDict
from numpy.typing import NDArray
from torch import embedding
from .search_utils import (
        CACHE_DIR,
        DATA_PATH,
        DEFAULT_CHUNK_OVERLAP,
        DEFAULT_CHUNK_SIZE,
        DEFAULT_SEMANTIC_CHUNCH_SIZE,
        MODEL,
        Movie,
        CACHE_EMBEDDINGS_PATH,
        load_movies,
        DEFAULT_SEARCH_LIMIT,
        DEFAULT_CHUNK_SIZE,
        CHUNK_EMBEDDINGS_PATH,
        CHUNK_METADATA_PATH,
        format_search_result
        )
import numpy as np
import os
import re
import json

class SemanticSearchResult(TypedDict):
    score: float
    title: str
    description: str

class ChunkMetadata(TypedDict):
    movie_idx: int
    chunk_idx: int
    total_chunks: int

class ChunkScoreData(TypedDict):
    chunk_idx: int
    movie_idx: int
    score: float

class MovieScores(TypedDict):
    idx: str
    score: float

EmbeddingArray = NDArray[Any]

class SemanticSearch:
    def __init__(self, model=MODEL) -> None:
        self.model = SentenceTransformer(MODEL)
        self.embeddings: EmbeddingArray | None = None
        self.documents: list[Movie] | None = None
        self.document_map: dict[int, Movie] = {}

    def generate_embedding(self, text: str) -> EmbeddingArray:
        if not text or len(text.strip()) == 0:
            raise ValueError("text error")
        return self.model.encode([text])[0]

    def build_embeddings(self, documents: list[Movie]) -> EmbeddingArray:
        self.documents = documents
        doc_list: list[str] = []

        for doc in documents:
            self.document_map[doc['id']] = doc
            doc_string = f"{doc['title']}: {doc['description']}"
            doc_list.append(doc_string)

        self.embeddings = self.model.encode(doc_list, show_progress_bar=True)
        np.save(CACHE_EMBEDDINGS_PATH, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[Movie]) -> EmbeddingArray:
        self.documents = documents
        doc_list = []

        for doc in documents:
            self.document_map[doc['id']] = doc
            doc_string = f"{doc['title']}: {doc['description']}"
            doc_list.append(doc_string)

        if os.path.exists(CACHE_EMBEDDINGS_PATH):
            self.embeddings = np.load(CACHE_EMBEDDINGS_PATH)
            if self.embeddings is not None and len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embeddings(documents)

    def search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[SemanticSearchResult] | None:
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call 'load_or_create_embeddings' first.")
        if self.documents is None:
            raise ValueError("No embeddings loaded. Call 'load_or_create_embeddings' first.")

        embedded_query = self.generate_embedding(query)

        cosine_similarities: list[tuple[float, Movie]] = []
        for i, embed in enumerate(self.embeddings):
            similarity_score = cosine_similarity(embedded_query, embed)
            cosine_similarities.append((similarity_score, self.documents[i]))


        sorted_scores = sorted(cosine_similarities, key=lambda x: x[0], reverse=True)

        top_list: list[SemanticSearchResult] = []
        for score, doc in cosine_similarities[:limit]:
            res: SemanticSearchResult = {'score':score, 'title':doc['title'], 'description':doc['description']}
            top_list.append(res)

        return top_list

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = MODEL) -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[Movie]) -> EmbeddingArray:
        self.documents = documents
        chunk_metadata: list[ChunkMetadata] = []
        all_chunks = []

        for doc in documents:
            self.document_map[doc['id']] = doc

        for doc in documents:
            if doc['description'] is None:
                continue

            description = doc['description']
            chunked_description = semantic_chunk(description, 4, 1)
            for idx, chunk in enumerate(chunked_description):
                all_chunks.append(chunk)
                chunk_metadata.append({
                        'movie_idx': doc['id'],
                        'chunk_idx': idx,
                        'total_chunks': len(chunked_description)
                    })

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        np.save(CHUNK_EMBEDDINGS_PATH, self.chunk_embeddings)

        with open(CHUNK_METADATA_PATH, "w") as f:
            json.dump({"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[Movie]) -> EmbeddingArray:
        self.documents = documents
        for doc in documents:
            self.document_map[doc['id']] = doc

        if (os.path.exists(CHUNK_EMBEDDINGS_PATH) and os.path.getsize(CHUNK_EMBEDDINGS_PATH) > 0
            and os.path.exists(CHUNK_METADATA_PATH) and os.path.getsize(CHUNK_METADATA_PATH) > 0):

            self.chunk_embeddings = np.load(CHUNK_EMBEDDINGS_PATH)

            with open(CHUNK_METADATA_PATH, 'r') as f:
                self.chunk_metadata = json.load(f)

            return self.chunk_embeddings

        else:
            return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10) -> list[dict]:
        if self.chunk_embeddings is None:
            raise ValueError("No embeddings loaded. Call 'load_or_create_embeddings' first.")
        if self.chunk_metadata is None:
            raise ValueError("No embeddings loaded. Call 'load_or_create_embeddings' first.")
        if self.documents is None:
            raise ValueError("No embeddings loaded. Call 'load_or_create_embeddings' first.")

        embedded_query = self.generate_embedding(query)
        chunk_scores: list[ChunkScoreData] = []

        for idx, chunk in enumerate(self.chunk_embeddings):
            similarity_score = cosine_similarity(embedded_query, chunk)
            movie_idx = self.chunk_metadata['chunks'][idx]['movie_idx']
            chunk_idx = self.chunk_metadata['chunks'][idx]['chunk_idx']
            chunk_scores.append({
                    'chunk_idx': chunk_idx,
                    'movie_idx': movie_idx,
                    'score': similarity_score
                })

        movie_scores = {}

        for chunk_score in chunk_scores:
            movie = chunk_score['movie_idx']
            if movie not in movie_scores or chunk_score['score'] > movie_scores[movie]['score']:
                movie_scores[movie] = chunk_score

        sorted_movies = sorted(movie_scores.values(), key=lambda x: x['score'], reverse=True)[:limit]

        res = []
        for movie in sorted_movies:
            document = self.document_map[movie['movie_idx']]
            title = document['title']
            res.append(format_search_result(document['id'], title, document['description'][:100], movie['score']))

        return res

def verify_model():
    ss = SemanticSearch()
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")

def embed_text(text:str) -> None:
    ss = SemanticSearch()
    embeddings = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embeddings[:3]}")
    print(f"Dimensions: {embeddings.shape[0]}")

def verify_embeddings():
    ss = SemanticSearch()
    movies = load_movies()
    embeddings = ss.load_or_create_embeddings(movies)
    print(f"Number of docs: {len(movies)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query:str) -> None:
    ss = SemanticSearch()
    embedded = ss.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedded[:3]}")
    print(f"Shape: {embedded.shape}")

def cosine_similarity(vec1: EmbeddingArray, vec2: EmbeddingArray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def search_query(query, limit: int = DEFAULT_SEARCH_LIMIT) -> None:
    ss = SemanticSearch()
    movies = load_movies()
    ss.load_or_create_embeddings(movies)
    search_res = ss.search(query, limit)

    if search_res is not None:
        for i, res in enumerate(search_res):
            print(f"{i + 1}. {res["title"]} (score: {res["score"]:.4f})")
            print(f"  {res["description"][:100]}...")
            print()

def chunk(text:str, size=200, overlap = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks = []

    n_words = len(words)
    i = 0
    while i < n_words - overlap:
        chunk_words = words[i: i + size]
        chunks.append(" ".join(chunk_words))
        i += size - overlap

    return chunks


def semantic_chunk(text:str, max_chunk_size:int, overlap:int) -> list[str]:
    text = text.strip()
    if text is None:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []

    n_sentences = len(sentences)

    if n_sentences == 1 and not sentences[0].endswith(('.', '?', '!')):
        return sentences

    i = 0
    while i < n_sentences - overlap:
        chunk_words = sentences[i: i + max_chunk_size]
        striped_words = []
        for word in chunk_words:
            if word.strip() is not None:
                striped_words.append(word.strip())

        chunks.append(" ".join(chunk_words))
        i += max_chunk_size - overlap



    return chunks

def chunk_text(text: str, size: int = DEFAULT_CHUNK_SIZE, overlap = DEFAULT_CHUNK_OVERLAP) -> None:
    chunks = chunk(text, size, overlap)
    print(f"Semantically chunking {len(text)} characters")
    for i, chnk in enumerate(chunks):
        print(f"{i + 1}. {chnk}")

def semantic_chunk_text(text: str, size: int = DEFAULT_CHUNK_SIZE, overlap = DEFAULT_CHUNK_OVERLAP) -> None:
    chunks = semantic_chunk(text, size, overlap)
    print(f"Semantically chunking {len(text)} characters")
    for i, chnk in enumerate(chunks):
        print(f"{i + 1}. {chnk}")

def embed_chunks() -> None:
    movies = load_movies()
    CSS = ChunkedSemanticSearch()
    embeddings = CSS.load_or_create_chunk_embeddings(movies)
    print(f"Generated {len(embeddings)} chunked embeddings")

def search_chunks(text:str, limit: int) -> None:
    movies = load_movies()
    CSS = ChunkedSemanticSearch()
    CSS.load_or_create_chunk_embeddings(movies)
    results = CSS.search_chunks(text, limit)
    for i, res in enumerate(results, start=1):
        print(f"\n{i}. {res['title']} (score: {res['score']:.4f})")
        print(f"    {res['document']}...")





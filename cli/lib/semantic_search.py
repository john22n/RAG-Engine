from sentence_transformers import SentenceTransformer
from torch import embedding
from .search_utils import (
        CACHE_DIR,
        DATA_PATH,
        MODEL,
        Movie,
        CACHE_EMBEDDINGS_PATH,
        load_movies
        )
import numpy as np
import os

class SemanticSearch:
    def __init__(self, model=MODEL) -> None:
        self.model = SentenceTransformer(MODEL)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text):
        if not text or len(text.strip()) == 0:
            raise ValueError("text error")
        return self.model.encode([text])[0]

    def build_embeddings(self, documents: list[Movie]):
        self.documents = documents
        doc_list = []

        for doc in documents:
            self.document_map[doc['id']] = doc
            doc_string = f"{doc['title']}: {doc['description']}"
            doc_list.append(doc_string)

        self.embeddings = self.model.encode(doc_list, show_progress_bar=True)
        np.save(CACHE_EMBEDDINGS_PATH, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list):
        self.documents = documents
        doc_list = []

        for doc in documents:
            self.document_map[doc['id']] = doc
            doc_string = f"{doc['title']}: {doc['description']}"
            doc_list.append(doc_string)

        if os.path.exists(CACHE_EMBEDDINGS_PATH):
            self.embeddings = np.load(CACHE_EMBEDDINGS_PATH)

        if self.embeddings is not None and len(documents) == len(self.embeddings):
            return self.embeddings
        else:
            return self.build_embeddings(documents)

    def search(self, query: str, limit: int) -> list | None:
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call 'load_or_create_embeddings' first.")

        embedded_query = self.generate_embedding(query)

        cosine_similarities = []

        if self.documents is not None:
            for doc, embed in zip(self.documents, self.embeddings):
                similarity_score = cosine_similarity(embedded_query, embed)
                cosine_similarities.append((similarity_score, doc))


        sorted_scores = sorted(cosine_similarities, key=lambda x: x[0], reverse=True)

        top_list = []

        for score in sorted_scores:
            res = {'score':score[0], 'title':score[1]['title'], 'description':score[1]['description']}
            top_list.append(res)
            if len(top_list) >= limit:
                return top_list

        return top_list

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

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)



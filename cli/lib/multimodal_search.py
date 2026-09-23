from PIL import Image
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel

from lib.search_utils import Movie, load_movies
from lib.semantic_search import EmbeddingArray, cosine_similarity

class MultimodalSearch:

    def __init__(self, documents: list[Movie], model_name="sentence-transformers/clip-ViT-B-32") -> None:
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.documents = documents
        self.texts: list[str] = []
        self.text_embeddings: EmbeddingArray

    def build_embeddings(self) -> None:
        for doc in self.documents:
            doc_string = f"{doc['title']}: {doc['description']}"
            self.texts.append(doc_string)

        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, image_url):
        image = Image.open(image_url)
        image_embedding = self.model.encode(image)
        return image_embedding

    def search_with_image(self, image_path: str) -> list[dict]:
        embedded_image = self.embed_image(image_path)
        doc_scores = []
        for i, embed in enumerate(self.text_embeddings):
            similarity_score = cosine_similarity(embed, embedded_image)
            doc = self.documents[i]
            doc_obj = {'score': similarity_score, 'title': doc['title'], 'description': doc['description']}
            doc_scores.append(doc_obj)

        sorted_scores = sorted(doc_scores, key=lambda x: x['score'], reverse=True)

        return sorted_scores[:10]






def verify_image_embeddings(image_path):
    movies = load_movies()
    mm_search = MultimodalSearch(movies)
    embedding = mm_search.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")

def image_search(image_path):
    movies = load_movies()
    mm_search = MultimodalSearch(movies)
    mm_search.build_embeddings()

    sorted_movies = mm_search.search_with_image(image_path)

    for i, mov in enumerate(sorted_movies, start=1):
        print(f"{i}. {mov['title']} (similarity: {mov['score']:.2f}\n {mov['description']}")




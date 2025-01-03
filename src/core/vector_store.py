from typing import List
import numpy as np
from numpy import ndarray
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class VectorStore:
    def __init__(self, embedding_model: str = "sentence-transformers/all-mpnet-base-v2"):
        self.embedding_model_name = embedding_model
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.documents: List[str] = []
        self.embeddings: List[np.ndarray] = []

    def add_documents(self, documents: List[str]):
        """Add documents and their embeddings to the store."""
        self.documents.extend(documents)

        # Compute embeddings in batches with progress bar
        batch_size = 5
        num_batches = (len(documents) + batch_size - 1) // batch_size

        with tqdm(total=len(documents), desc="Storing Vectors") as pbar:
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                embeddings = self._get_embeddings(batch)
                self.embeddings.extend(embeddings)
                pbar.update(len(batch))

    def _get_embeddings(self, texts: List[str]) -> ndarray:
        """Get embeddings using SentenceTransformer."""
        return self.embedding_model.encode(texts, convert_to_numpy=True)

    def find_relevant_context(self, query: str, num_chunks: int = 6) -> str:
        """Find most relevant document chunks for a query."""
        query_embedding = self._get_embeddings([query])[0]

        similarities = []
        for doc_embedding in self.embeddings:
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            similarities.append(similarity)

        top_indices = np.argsort(similarities)[-num_chunks:]
        context = " ".join([self.documents[i] for i in top_indices])

        return context

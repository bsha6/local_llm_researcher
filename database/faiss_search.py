import numpy as np
import sqlite3

from database.faiss_index import FaissIndex
from data_pipeline.generate_embeddings import E5Embedder
from utils.file_operations import load_config
config = load_config()
DB_NAME = config["database"]["arxiv_db_path"]

class FaissSearcher:
    def __init__(self, faiss_index: FaissIndex, db=DB_NAME):
        """
        Initialize the FAISS searcher.

        :param faiss_index: An instance of the FaissIndex class.
        :param db: Path to SQLite database storing metadata.
        """
        self.faiss_index = faiss_index
        self.db = db

    def search(self, query_embedding: np.ndarray, top_k=5):
        """
        Search for nearest embeddings in FAISS.

        :param query_embedding: NumPy array of shape (1, dim)
        :param top_k: Number of results to retrieve.
        :return: List of retrieved text chunks.
        """
        assert query_embedding.shape[1] == self.faiss_index.dim, "Query embedding dimension mismatch!"

        distances, indices = self.faiss_index.index.search(query_embedding, top_k)
        return self._fetch_metadata(indices[0])

    def _fetch_metadata(self, indices):
        """
        Retrieve text chunks from SQLite.

        :param indices: List of FAISS indices.
        :return: List of text chunks.
        """
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()
        retrieved_chunks = []

        for idx in indices:
            cursor.execute("SELECT chunk_text FROM paper_chunks WHERE faiss_idx=?", (idx+1,))
            result = cursor.fetchone()
            if result:
                retrieved_chunks.append(result[0])

        conn.close()
        return retrieved_chunks

if __name__ == "__main__":
    # Initialize the embedding model
    embedder = E5Embedder()
    
    # Create a test query
    test_query = "Chain of thought reasoning"
    
    # Generate embedding for the test query
    query_embedding = embedder.generate_embeddings([test_query], mode="query")
    
    # Initialize FAISS index and searcher
    faiss_index = FaissIndex()
    faiss_searcher = FaissSearcher(faiss_index)
    
    # Search for similar chunks
    print(f"\nSearching for chunks similar to: '{test_query}'")
    print("-" * 80)
    
    # Get both chunk IDs and distances
    chunk_ids, distances = faiss_index.search(query_embedding, top_k=5)
    
    # Get the actual chunk texts
    chunk_texts = faiss_index.get_chunk_texts(chunk_ids)
    
    # Print results
    for i, (chunk_id, paper_id, chunk_text) in enumerate(chunk_texts):
        print(f"\nResult {i+1} (Distance: {distances[i]:.4f})")
        print(f"Paper ID: {paper_id}")
        print(f"Chunk ID: {chunk_id}")
        print(f"Text: {chunk_text[:300]}...")
        print("-" * 80)

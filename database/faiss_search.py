import numpy as np
import sqlite3

from database.faiss_index import FaissIndex
from utils.file_operations import load_config
config = load_config()
DB_NAME = config["database"]["arxiv_db_path"]

class FaissSearcher:
    def __init__(self, faiss_index: FaissIndex, metadata_db=DB_NAME):
        """
        Initialize the FAISS searcher.

        :param faiss_index: An instance of the FaissIndex class.
        :param metadata_db: Path to SQLite database storing metadata.
        """
        self.faiss_index = faiss_index
        self.metadata_db = metadata_db

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
        conn = sqlite3.connect(self.metadata_db)
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
    config = load_config()
    dim = config["models"]["e5_small"]["dimensions"]
    # Example Usage
    # query_embedding = np.random.rand(1, dim).astype(np.float32)  # Replace with actual query embedding
    # faiss_index = FaissIndex()
    # faiss_searcher = FaissSearcher(faiss_index)
    # retrieved_texts = faiss_searcher.search(query_embedding)

    # print("Top Matching Chunks:")
    # for text in retrieved_texts:
    #     print(text)

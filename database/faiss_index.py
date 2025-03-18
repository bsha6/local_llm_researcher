import faiss
import numpy as np

from utils.file_operations import load_config
from database.sqlite_db import DatabaseManager

# Load configuration
config = load_config()
ARXIV_DB_PATH = config["database"]["arxiv_db_path"]
DIM = config["models"]["e5_small"]["dimensions"]

class FaissIndex:
    def __init__(self, index_path="faiss_index.idx", db_path=ARXIV_DB_PATH, dim=DIM, 
                 use_hnsw=True, M=32, efConstruction=100, efSearch=40, batch_size=1000):
        """
        Initialize FAISS index with optimized HNSW for fast approximate nearest neighbor search.
        
        :param index_path: Path to store FAISS index.
        :param db_path: Path to SQLite database.
        :param dim: Embedding dimension (E5-small outputs 384D).
        :param use_hnsw: Whether to use HNSW (recommended for local + fast queries).
        :param M: Number of connections per layer in HNSW graph (higher = better recall but more memory).
        :param efConstruction: Size of the dynamic candidate list for construction (higher = better quality but slower build).
        :param efSearch: Size of the dynamic candidate list for search (higher = better recall but slower search).
        :param batch_size: Number of records to write in a single batch.
        """
        self.index_path = index_path
        self.db_path = db_path
        self.dim = dim
        self.use_hnsw = use_hnsw
        self.M = M
        self.efConstruction = efConstruction
        self.efSearch = efSearch
        self.batch_size = batch_size
        self.index = self._initialize_index()

    def _initialize_index(self):
        """Creates or loads a FAISS index with optimized HNSW parameters."""
        try:
            index = faiss.read_index(self.index_path)
            if isinstance(index, faiss.IndexHNSWFlat):
                index.hnsw.efSearch = self.efSearch
            print(f"Loaded existing FAISS index with {index.ntotal} vectors.")
            return index
        except Exception as e:
            print(f"Could not load index: {e}")
            print("Creating a new FAISS index...")
            if self.use_hnsw:
                # Initialize HNSW index with optimized parameters
                index = faiss.IndexHNSWFlat(self.dim, self.M)
                index.hnsw.efConstruction = self.efConstruction
                index.hnsw.efSearch = self.efSearch
            else:
                index = faiss.IndexFlatL2(self.dim)  # Brute-force search

            return index


    def add_embeddings(self, embeddings: np.ndarray, chunk_ids: list):
        """
        Adds embeddings to the FAISS index and updates the paper_chunks table with FAISS indices.
        
        :param embeddings: Numpy array of embeddings to add
        :param chunk_ids: List of chunk IDs from SQLite corresponding to each embedding
        """
        assert embeddings.shape[0] == len(chunk_ids), "Number of embeddings must match number of chunk IDs"
        assert embeddings.shape[1] == self.dim, "Embedding dimension mismatch!"
        
        # Get the current index size before adding
        start_idx = self.index.ntotal
        
        # Add embeddings to FAISS index
        self.index.add(embeddings)
        
        # Prepare batch updates for the database
        updates = [(start_idx + i, chunk_id) for i, chunk_id in enumerate(chunk_ids)]
        
        # Update the paper_chunks table in batches
        with DatabaseManager(self.db_path) as cursor:
            for i in range(0, len(updates), self.batch_size):
                batch = updates[i:i + self.batch_size]
                cursor.executemany(
                    'UPDATE paper_chunks SET faiss_idx = ? WHERE chunk_id = ?',
                    batch
                )
        
        self.save_index()
        
        print(f"Added {len(chunk_ids)} embeddings with corresponding chunk IDs")

    def save_index(self):
        """Saves FAISS index to disk."""
        faiss.write_index(self.index, self.index_path)
        print(f"FAISS index saved to {self.index_path}")

    def search(self, query_embedding: np.ndarray, top_k=5):
        """
        Search FAISS index for nearest neighbors and return their chunk IDs.
        
        :param query_embedding: Embedding vector to search for
        :param top_k: Number of results to return
        :return: Tuple of (chunk_ids, distances)
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        assert query_embedding.shape[1] == self.dim, "Query embedding dimension mismatch!"
        
        # Ensure we don't request more results than we have
        actual_top_k = min(top_k, self.index.ntotal)
        if actual_top_k < top_k:
            print(f"Warning: Requested {top_k} results but only {actual_top_k} vectors in index")
        
        # Search the FAISS index
        distances, faiss_indices = self.index.search(query_embedding, actual_top_k)
        
        # Convert indices to integers for SQL query
        faiss_indices_list = [int(idx) for idx in faiss_indices[0]]
        
        chunk_ids = []
        if faiss_indices_list:
            # Use placeholders for the IN clause
            placeholders = ','.join(['?' for _ in faiss_indices_list])
            query = f'''
                SELECT faiss_idx, chunk_id 
                FROM paper_chunks 
                WHERE faiss_idx IN ({placeholders})
                ORDER BY CASE faiss_idx 
            '''
            
            # Add custom ordering to match the FAISS results order
            for i, idx in enumerate(faiss_indices_list):
                query += f" WHEN {idx} THEN {i} "
            query += " END"
            
            with DatabaseManager(self.db_path) as cursor:
                cursor.execute(query, faiss_indices_list)
                results = cursor.fetchall()
                
                # Create a mapping from faiss_idx to chunk_id
                idx_to_chunk = {idx: chunk_id for idx, chunk_id in results}
                
                # Preserve the original order from FAISS search
                for idx in faiss_indices[0]:
                    chunk_ids.append(idx_to_chunk.get(int(idx)))
        
        return chunk_ids, distances[0]

    def reset_faiss_indices(self):
        """Reset all FAISS indices in the paper_chunks table."""
        with DatabaseManager(self.db_path) as cursor:
            cursor.execute('UPDATE paper_chunks SET faiss_idx = NULL')
        self.index.reset()
        print("FAISS index and paper_chunks.faiss_idx values reset")
    
    def get_chunk_texts(self, chunk_ids):
        """
        Retrieve the text content for the given chunk IDs.
        
        :param chunk_ids: List of chunk IDs to retrieve
        :return: List of (chunk_id, paper_id, chunk_text) tuples
        """
        if not chunk_ids:
            return []
            
        placeholders = ','.join(['?' for _ in chunk_ids])
        query = f'''
            SELECT chunk_id, paper_id, chunk_text 
            FROM paper_chunks 
            WHERE chunk_id IN ({placeholders})
            ORDER BY CASE chunk_id 
        '''
        
        # Add custom ordering to match the input order
        for i, chunk_id in enumerate(chunk_ids):
            query += f" WHEN {chunk_id} THEN {i} "
        query += " END"
        
        with DatabaseManager(self.db_path) as cursor:
            cursor.execute(query, chunk_ids)
            return cursor.fetchall()

    def rebuild_index_from_db(self, embedding_model):
        """
        Rebuild the FAISS index from the database using the provided embedding model.
        
        :param embedding_model: A function or model that converts text to embeddings
        """
        # Reset the index
        self.index.reset()
        
        with DatabaseManager(self.db_path) as cursor:
            # Check if the paper_chunks table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='paper_chunks'")
            if not cursor.fetchone():
                print("Error: paper_chunks table does not exist")
                return
            
            # Get all chunks that need embeddings
            cursor.execute("SELECT COUNT(*) FROM paper_chunks")
            total_chunks = cursor.fetchone()[0]
            
            if total_chunks == 0:
                print("No chunks found in the database")
                return
                
            print(f"Found {total_chunks} chunks in database. Regenerating embeddings and rebuilding FAISS index...")
            
            # Get all chunks
            cursor.execute("SELECT chunk_id, chunk_text FROM paper_chunks")
            chunks = cursor.fetchall()
            
            # Process in batches
            for i in range(0, len(chunks), self.batch_size):
                batch = chunks[i:i+self.batch_size]
                
                chunk_ids = [chunk[0] for chunk in batch]
                chunk_texts = [chunk[1] for chunk in batch]
                
                # Generate embeddings for this batch
                embeddings = np.array([embedding_model(text) for text in chunk_texts]).astype("float32")
                
                # Add to FAISS index and update database
                self.add_embeddings(embeddings, chunk_ids)
                
                print(f"Processed {min(i + self.batch_size, total_chunks)}/{total_chunks} chunks")
            
            self.save_index()
            print(f"FAISS index rebuilt with {self.index.ntotal} vectors")
    
    def insert_chunks_into_db(self, chunks, paper_id, embedding_model):
        """
        Inserts chunked text into SQLite and stores embeddings in FAISS.

        :param chunks: List of text chunks.
        :param paper_id: The paper ID these chunks belong to.
        :param embedding_model: A function or model for generating embeddings.
        """
        if not chunks:  # Handle empty list case
            return

        chunk_ids = []

        with DatabaseManager(self.db_path) as cursor:
            for chunk in chunks:
                cursor.execute(
                    "INSERT INTO paper_chunks (paper_id, chunk_text) VALUES (?, ?)",
                    (paper_id, chunk)
                )
                chunk_id = cursor.lastrowid
                chunk_ids.append(chunk_id)

        if chunk_ids:
            # Generate embeddings for all chunks at once to ensure consistent shapes
            embeddings = embedding_model(chunks)
            
            # Ensure embeddings are in the correct shape
            if isinstance(embeddings, list):
                embeddings = np.array(embeddings)
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            
            # Convert to float32 for FAISS
            embeddings = embeddings.astype("float32")
            
            # Verify dimensions
            if embeddings.shape[1] != self.dim:
                raise ValueError(f"Embedding dimension mismatch. Expected {self.dim}, got {embeddings.shape[1]}")
            
            # Insert embeddings into FAISS
            self.add_embeddings(embeddings, chunk_ids)


if __name__ == "__main__":    
    print("ran")

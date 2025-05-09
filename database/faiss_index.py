import faiss
import numpy as np
import os
from typing import List, Tuple, Optional
from pathlib import Path
from utils.file_operations import load_config
from database.sqlite_db import DatabaseManager

class FaissIndex:
    _config = None

    @classmethod
    def get_config(cls):
        if cls._config is None:
            cls._config = load_config()
        return cls._config

    def __init__(self, index_path: Optional[str] = None, db_path=None, dim=None, 
                 M=16, efConstruction=40, efSearch=16, flat_batch_size=200, hnsw_batch_size=10):
        """
        Initialize FAISS index with support for both FlatL2 and HNSW indices.
        
        :param index_path: Path to store FAISS index.
        :param db_path: Path to SQLite database.
        :param dim: Embedding dimension (E5-small outputs 384D).
        :param M: Number of connections per layer in HNSW graph (higher = better recall but more memory).
        :param efConstruction: Size of the dynamic candidate list for construction (higher = better quality but slower build).
        :param efSearch: Size of the dynamic candidate list for search (higher = better recall but slower search).
        :param flat_batch_size: Batch size for FlatL2 operations (can be larger for faster processing).
        :param hnsw_batch_size: Batch size for HNSW operations (should be smaller for stability).
        """
        config = self.get_config()
        self.index_path = index_path or config["database"].get("faiss_index_path", "faiss_index.idx")
        self.db_path = db_path or config["database"]["arxiv_db_path"]
        self.dim = dim or config["models"]["e5_small"]["dimensions"]
        self.M = M
        self.efConstruction = efConstruction
        self.efSearch = efSearch
        self.flat_batch_size = flat_batch_size
        self.hnsw_batch_size = hnsw_batch_size
        self.is_hnsw = False
        
        self.index = self._initialize_index()

    def _initialize_index(self):
        """Creates or loads a FAISS index."""
        try:
            if os.path.exists(self.index_path):
                index = faiss.read_index(self.index_path)
                self.is_hnsw = isinstance(index, faiss.IndexHNSWFlat)
                if self.is_hnsw:
                    index.hnsw.efSearch = self.efSearch
                print(f"Loaded existing {'HNSW' if self.is_hnsw else 'FlatL2'} index with {index.ntotal} vectors")
                return index
        except Exception as e:
            print(f"Could not load index: {e}")
        
        print("Creating new FlatL2 index for building phase...")
        # Start with FlatL2 for stable building
        index = faiss.IndexFlatL2(self.dim)
        self.is_hnsw = False
        
        # Set thread settings for better stability
        faiss.omp_set_num_threads(4)
        return index

    def convert_to_hnsw(self):
        """Convert FlatL2 index to HNSW for faster searching."""
        if self.is_hnsw:
            print("Index is already an HNSW index")
            return
            
        if not isinstance(self.index, faiss.IndexFlatL2):
            print("Index is not suitable for conversion")
            return
            
        print("\nConverting FlatL2 to HNSW index...")
        print(f"Current index contains {self.index.ntotal} vectors")
        
        try:
            # Create new HNSW index
            hnsw_index = faiss.IndexHNSWFlat(self.dim, self.M)
            hnsw_index.hnsw.efConstruction = self.efConstruction
            hnsw_index.hnsw.efSearch = self.efSearch
            
            total_vectors = self.index.ntotal
            
            # Process in small batches for stability
            for i in range(0, total_vectors, self.hnsw_batch_size):
                end_idx = min(i + self.hnsw_batch_size, total_vectors)
                # Get batch of vectors
                vectors = np.zeros(((end_idx - i), self.dim), dtype=np.float32)
                self.index.reconstruct_n(i, end_idx - i, vectors)
                
                # Add to HNSW index
                hnsw_index.add(vectors)
                print(f"Converted vectors {i} to {end_idx} of {total_vectors}")
            
            # Replace the index
            self.index = hnsw_index
            self.is_hnsw = True
            
            # Save the converted index
            self.save_index()
            print(f"Successfully converted to HNSW index with {self.index.ntotal} vectors")
            
        except Exception as e:
            print(f"Error during conversion: {e}")
            print("Keeping original FlatL2 index")

    def add_embeddings(self, embeddings: np.ndarray, chunk_ids: list):
        """
        Adds embeddings to the FAISS index and updates the paper_chunks table.
        Uses appropriate batch size based on index type.
        
        :param embeddings: Numpy array of embeddings to add
        :param chunk_ids: List of chunk IDs from SQLite corresponding to each embedding
        """
        assert embeddings.shape[0] == len(chunk_ids), "Number of embeddings must match number of chunk IDs"
        assert embeddings.shape[1] == self.dim, "Embedding dimension mismatch!"
        
        # Get the current index size before adding
        start_idx = self.index.ntotal
        
        # Use appropriate batch size based on index type
        batch_size = self.hnsw_batch_size if self.is_hnsw else self.flat_batch_size
        
        # Process in batches
        for i in range(0, len(embeddings), batch_size):
            batch_end = min(i + batch_size, len(embeddings))
            batch_embeddings = embeddings[i:batch_end]
            batch_chunk_ids = chunk_ids[i:batch_end]
            
            # Add embeddings to FAISS index
            self.index.add(batch_embeddings)
            
            # Prepare updates for this batch
            updates = [(start_idx + j, chunk_id) for j, chunk_id in enumerate(batch_chunk_ids, start=i)]
            
            # Update the paper_chunks table
            with DatabaseManager(self.db_path) as cursor:
                cursor.executemany(
                    'UPDATE paper_chunks SET faiss_idx = ? WHERE chunk_id = ?',
                    updates
                )
            
            # Save after each batch for safety
            self.save_index()
            print(f"Added {len(batch_chunk_ids)} embeddings with corresponding chunk IDs")

    def save_index(self):
        """Saves FAISS index to disk."""
        faiss.write_index(self.index, self.index_path)
        print(f"FAISS index saved to {self.index_path}")

    def search(self, query_embedding: np.ndarray, top_k=5):
        """
        Search FAISS index for nearest neighbors and return their chunk IDs.
        Automatically converts to HNSW if still in building mode.
        
        :param query_embedding: Embedding vector to search for
        :param top_k: Number of results to return
        :return: Tuple of (chunk_ids, distances)
        """
        # Convert to HNSW if we're still in building mode
        if isinstance(self.index, faiss.IndexFlatL2):
            self.convert_to_hnsw()
        
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

    def reset(self):
        """Reset to a fresh FlatL2 index for rebuilding."""
        self.index = faiss.IndexFlatL2(self.dim)
        self.is_hnsw = False
        with DatabaseManager(self.db_path) as cursor:
            cursor.execute('UPDATE paper_chunks SET faiss_idx = NULL')
        print("Index reset to FlatL2 and paper_chunks.faiss_idx values cleared")

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
            for i in range(0, len(chunks), self.flat_batch_size):
                batch = chunks[i:i+self.flat_batch_size]
                
                chunk_ids = [chunk[0] for chunk in batch]
                chunk_texts = [chunk[1] for chunk in batch]
                
                # Generate embeddings for this batch
                embeddings = np.array([embedding_model(text) for text in chunk_texts]).astype("float32")
                
                # Add to FAISS index and update database
                self.add_embeddings(embeddings, chunk_ids)
                
                print(f"Processed {min(i + self.flat_batch_size, total_chunks)}/{total_chunks} chunks")
            
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

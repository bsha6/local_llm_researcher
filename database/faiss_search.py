import numpy as np
import sqlite3
import argparse
import os
import gc
import torch
from typing import List, Tuple, Optional, Dict

from database.faiss_index import FaissIndex
from data_pipeline.generate_embeddings import E5Embedder
from utils.file_operations import load_config
from database.sqlite_db import DatabaseManager

class FaissSearcher:
    _config = None

    @classmethod
    def get_config(cls):
        if cls._config is None:
            cls._config = load_config()
        return cls._config

    def __init__(self, faiss_index: Optional[FaissIndex] = None):
        """
        Initialize the FAISS searcher.

        :param faiss_index: An instance of the FaissIndex class.
        """
        self.config = self.get_config()
        self.faiss_index = faiss_index or FaissIndex()
        self.db = self.config["database"]["arxiv_db_path"]

    def rebuild_index(self, embedding_function):
        """
        Rebuild the FAISS index using the provided embedding function.
        Uses a two-phase approach: build with FlatL2, then convert to HNSW for searching.
        
        :param embedding_function: Function that takes text and returns embeddings
        :return: None
        """
        print("Starting FAISS index rebuild...")
        
        # Delete existing index file if it exists
        if os.path.exists(self.faiss_index.index_path):
            print(f"Removing existing index file: {self.faiss_index.index_path}")
            os.remove(self.faiss_index.index_path)
        
        # Reset to a fresh FlatL2 index
        self.faiss_index.reset()
        gc.collect()  # Force garbage collection
        torch.cuda.empty_cache()  # Clear CUDA cache if available
        
        try:
            with DatabaseManager(self.db) as cursor:
                # Get total count first
                cursor.execute("SELECT COUNT(*) FROM paper_chunks")
                total_chunks = cursor.fetchone()[0]
                
                if total_chunks == 0:
                    print("No chunks found in database")
                    return
                
                print(f"Found {total_chunks} chunks in database")
                print("Building with FlatL2 index for stability...")
                
                # Process in larger batches for FlatL2
                batch_size = 200  # Larger batches for faster FlatL2 processing
                processed = 0
                
                while processed < total_chunks:
                    # Clear memory from previous batch
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Get next batch
                    cursor.execute("""
                        SELECT chunk_id, chunk_text 
                        FROM paper_chunks 
                        LIMIT ? OFFSET ?
                    """, (batch_size, processed))
                    
                    batch = cursor.fetchall()
                    if not batch:
                        break
                    
                    try:
                        # Process batch
                        chunk_ids = [chunk[0] for chunk in batch]
                        chunk_texts = [chunk[1] for chunk in batch]
                        
                        # Generate embeddings for batch
                        embeddings = embedding_function(chunk_texts)
                        
                        # Ensure correct shape
                        if isinstance(embeddings, list):
                            embeddings = np.array(embeddings)
                        if embeddings.ndim == 1:
                            embeddings = embeddings.reshape(1, -1)
                        
                        # Convert to float32 and ensure contiguous
                        embeddings = np.ascontiguousarray(embeddings.astype('float32'))
                        
                        # Add to FAISS index
                        self.faiss_index.add_embeddings(embeddings, chunk_ids)
                        
                        processed += len(batch)
                        print(f"Processed {processed}/{total_chunks} chunks")
                        
                        # Clear embeddings from memory
                        del embeddings
                        gc.collect()
                            
                    except Exception as e:
                        print(f"Error processing batch: {e}")
                        processed += len(batch)  # Skip problematic batch
                        continue
                    
                print("\nFlatL2 build complete!")
                print(f"Index contains {self.faiss_index.index.ntotal} vectors")
                
                # Convert to HNSW for faster searching
                print("\nConverting to HNSW index for faster searching...")
                self.faiss_index.convert_to_hnsw()
                
        except Exception as e:
            print(f"Error during rebuild: {e}")
            raise

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
    # Set up argument parser
    parser = argparse.ArgumentParser(description='FAISS search with optional index rebuilding')
    parser.add_argument('--rebuild', action='store_true', help='Rebuild the FAISS index')
    args = parser.parse_args()
    
    try:
        # Initialize components
        embedder = E5Embedder()
        faiss_index = FaissIndex()
        faiss_searcher = FaissSearcher(faiss_index)
        
        # Rebuild index if specified
        if args.rebuild:
            faiss_searcher.rebuild_index(lambda text: embedder.generate_embeddings(text, mode="query"))
        
        # Create a test query
        test_query = "Chain of thought reasoning"
        query_embedding = embedder.generate_embeddings([test_query], mode="query")
        
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
            
    except Exception as e:
        print(f"Error: {e}")
        raise

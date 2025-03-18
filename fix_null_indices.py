import numpy as np
import gc
import faiss
from database.faiss_index import FaissIndex
from data_pipeline.generate_embeddings import E5Embedder
from utils.file_operations import load_config
from database.sqlite_db import DatabaseManager

def create_optimized_hnsw_index(dim):
    """
    Create an HNSW index with optimized parameters for better stability.
    """
    # Create HNSW index with optimized parameters
    M = 16  # Number of connections per layer (default is 32)
    efConstruction = 40  # Size of the dynamic candidate list for construction (default is 40)
    index = faiss.IndexHNSWFlat(dim, M)
    index.hnsw.efConstruction = efConstruction
    index.hnsw.efSearch = 16  # Controls accuracy vs speed during search
    
    # Set thread settings for better stability
    faiss.omp_set_num_threads(4)  # Limit number of threads
    
    return index

def fix_null_indices():
    # Initialize components
    config = load_config()
    embedder = E5Embedder()
    
    print("Fetching chunks with NULL faiss_idx...")
    with DatabaseManager(config["database"]["arxiv_db_path"]) as cursor:
        # Get chunks with NULL faiss_idx
        cursor.execute("""
            SELECT chunk_id, chunk_text 
            FROM paper_chunks 
            WHERE faiss_idx IS NULL
        """)
        null_chunks = cursor.fetchall()
        
        if not null_chunks:
            print("No NULL faiss_idx values found.")
            return
            
        print(f"Found {len(null_chunks)} chunks with NULL faiss_idx")
        
        # Initialize FAISS index with optimized HNSW
        faiss_index = FaissIndex(use_hnsw=True)
        
        # Replace the default index with our optimized one
        if hasattr(faiss_index, 'index'):
            old_index = faiss_index.index
            faiss_index.index = create_optimized_hnsw_index(faiss_index.dim)
            if old_index is not None and old_index.ntotal > 0:
                # Copy existing vectors if any
                faiss.copy_index_data(old_index, faiss_index.index)
        
        # Process chunks in batches
        batch_size = 5  # Keep small batch size for stability
        total_batches = (len(null_chunks) - 1) // batch_size + 1
        
        for i in range(0, len(null_chunks), batch_size):
            batch = null_chunks[i:min(i + batch_size, len(null_chunks))]
            current_batch = i // batch_size + 1
            
            chunk_ids = [chunk[0] for chunk in batch]
            texts = [chunk[1] for chunk in batch]
            
            print(f"\nProcessing batch {current_batch}/{total_batches}")
            print(f"Generating embeddings for {len(texts)} chunks...")
            
            try:
                # Clear memory before processing new batch
                gc.collect()
                
                # Generate embeddings
                embeddings = embedder.generate_embeddings(texts)
                
                # Ensure embeddings are in the correct format
                if isinstance(embeddings, list):
                    embeddings = np.array(embeddings)
                if embeddings.ndim == 1:
                    embeddings = embeddings.reshape(1, -1)
                embeddings = embeddings.astype('float32')
                
                # Verify embedding dimensions
                if embeddings.shape[1] != faiss_index.dim:
                    raise ValueError(f"Embedding dimension mismatch. Expected {faiss_index.dim}, got {embeddings.shape[1]}")
                
                # Add to FAISS index
                print("Adding embeddings to FAISS index...")
                faiss_index.add_embeddings(embeddings, chunk_ids)
                
                # Save index after each batch
                try:
                    print("Saving index checkpoint...")
                    # Ensure the index is not being modified during save
                    with faiss.ThreadPool(1):  # Force single thread during save
                        faiss_index.save_index()
                except Exception as save_error:
                    print(f"Warning: Failed to save index checkpoint: {save_error}")
                    print("Continuing with next batch...")
                
                print(f"Successfully processed batch {current_batch}")
                
                # Clear embeddings from memory
                del embeddings
                gc.collect()
                
            except Exception as e:
                print(f"Error processing batch {current_batch}: {e}")
                print("Skipping to next batch...")
                continue
    
    print("\nFinished processing all chunks")
    
    # Final verification
    print("Verifying all chunks have been processed...")
    with DatabaseManager(config["database"]["arxiv_db_path"]) as cursor:
        cursor.execute("SELECT COUNT(*) FROM paper_chunks WHERE faiss_idx IS NULL")
        remaining_null = cursor.fetchone()[0]
        if remaining_null > 0:
            print(f"Warning: {remaining_null} chunks still have NULL faiss_idx values")
        else:
            print("All chunks have been successfully processed")
        
        # Print index statistics
        print("\nFAISS Index Statistics:")
        print(f"Total vectors: {faiss_index.index.ntotal}")
        print(f"Dimension: {faiss_index.dim}")

if __name__ == "__main__":
    try:
        fix_null_indices()
    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        # Force garbage collection
        gc.collect() 
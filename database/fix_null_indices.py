import numpy as np
from database.faiss_index import FaissIndex
from data_pipeline.generate_embeddings import E5Embedder
from utils.file_operations import load_config
from database.sqlite_db import DatabaseManager

def fix_null_indices():
    # Initialize components
    config = load_config()
    faiss_index = FaissIndex()
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
        
        # Process chunks in batches
        batch_size = 10
        for i in range(0, len(null_chunks), batch_size):
            batch = null_chunks[i:min(i + batch_size, len(null_chunks))]
            
            chunk_ids = [chunk[0] for chunk in batch]
            texts = [chunk[1] for chunk in batch]
            
            print(f"\nProcessing batch {i//batch_size + 1}/{(len(null_chunks)-1)//batch_size + 1}")
            print(f"Generating embeddings for {len(texts)} chunks...")
            
            # Generate embeddings
            try:
                embeddings = embedder.generate_embeddings(texts)
                
                # Ensure embeddings are in the correct format
                if isinstance(embeddings, list):
                    embeddings = np.array(embeddings)
                if embeddings.ndim == 1:
                    embeddings = embeddings.reshape(1, -1)
                embeddings = embeddings.astype('float32')
                
                # Add to FAISS index
                print("Adding embeddings to FAISS index...")
                faiss_index.add_embeddings(embeddings, chunk_ids)
                
                print(f"Successfully processed batch {i//batch_size + 1}")
            except Exception as e:
                print(f"Error processing batch: {e}")
                continue
    
    print("\nFinished processing all chunks")
    print("Saving FAISS index...")
    faiss_index.save_index()

if __name__ == "__main__":
    fix_null_indices() 
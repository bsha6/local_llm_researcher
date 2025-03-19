from typing import List, Tuple
import requests
import numpy as np
import logging

from database.faiss_search import FaissSearcher
from database.faiss_index import FaissIndex
from data_pipeline.generate_embeddings import E5Embedder
from utils.file_operations import load_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = load_config()

class RAGPipeline:
    def __init__(self, 
                 faiss_searcher: FaissSearcher = None,
                 embedder: E5Embedder = None,
                 ollama_url: str = "http://localhost:11434",
                 model_name: str = "llama3.2:3b",
                 num_chunks: int = 3,
                 temperature: float = 0.7):
        """
        Initialize the RAG pipeline with FAISS search and Ollama model.
        
        :param faiss_searcher: Instance of FaissSearcher for retrieval
        :param embedder: Instance of E5Embedder for query embedding
        :param ollama_url: URL for Ollama API
        :param model_name: Name of the Llama model to use
        :param num_chunks: Number of chunks to retrieve
        :param temperature: Temperature for LLM generation
        """
        self.faiss_searcher = faiss_searcher or FaissSearcher(FaissIndex())
        self.embedder = embedder or E5Embedder()
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.num_chunks = num_chunks
        self.temperature = temperature
        
        # Verify FAISS index is properly loaded
        if self.faiss_searcher.faiss_index.index.ntotal == 0:
            logger.warning("FAISS index is empty! Make sure it's properly initialized.")

    def retrieve_relevant_chunks(self, query: str) -> List[Tuple[str, float]]:
        """
        Retrieve relevant chunks using FAISS search.
        
        :param query: User query
        :return: List of (chunk_text, relevance_score) tuples
        """
        try:
            # Generate query embedding
            logger.info(f"Generating embedding for query: {query[:50]}...")
            query_embedding = self.embedder.generate_embeddings([query], mode="query")
            
            # Ensure embedding is the right shape and type
            if isinstance(query_embedding, list):
                query_embedding = np.array(query_embedding)
            
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
                
            query_embedding = query_embedding.astype(np.float32)
            
            logger.info(f"Embedding shape: {query_embedding.shape}")
            
            if query_embedding.shape[1] != self.faiss_searcher.faiss_index.dim:
                raise ValueError(
                    f"Embedding dimension mismatch. Expected {self.faiss_searcher.faiss_index.dim}, "
                    f"got {query_embedding.shape[1]}"
                )
            
            # Get chunk IDs and distances from FAISS
            logger.info("Performing FAISS search...")
            chunk_ids, distances = self.faiss_searcher.faiss_index.search(
                query_embedding, 
                top_k=min(self.num_chunks, self.faiss_searcher.faiss_index.index.ntotal)
            )
            
            if not chunk_ids:
                logger.warning("No chunks found in FAISS search")
                return []
            
            # Get the actual chunk texts
            logger.info("Retrieving chunk texts...")
            chunk_texts = self.faiss_searcher.faiss_index.get_chunk_texts(chunk_ids)
            
            # Combine texts with their relevance scores (1 / distance)
            results = []
            for i, (chunk_id, paper_id, chunk_text) in enumerate(chunk_texts):
                if chunk_text:  # Ensure we have valid text
                    relevance = 1.0 / (1.0 + distances[i])  # Convert distance to similarity score
                    results.append((chunk_text, relevance))
                    logger.debug(f"Added chunk {i+1} with relevance {relevance:.3f}")
            
            logger.info(f"Successfully retrieved {len(results)} relevant chunks")
            return results
            
        except Exception as e:
            logger.error(f"Error in retrieve_relevant_chunks: {str(e)}", exc_info=True)
            return []

    def generate_prompt(self, query: str, context_chunks: List[Tuple[str, float]]) -> str:
        """
        Generate a prompt for Llama using retrieved context.
        
        :param query: User query
        :param context_chunks: List of (chunk_text, relevance_score) tuples
        :return: Formatted prompt
        """
        try:
            # Sort chunks by relevance score
            sorted_chunks = sorted(context_chunks, key=lambda x: x[1], reverse=True)
            
            # Format context string with the most relevant chunks
            context = "\n\n".join([
                f"Relevant text {i+1} (relevance: {score:.2f}):\n{text}"
                for i, (text, score) in enumerate(sorted_chunks)
            ])
            
            # Create a structured prompt
            prompt = f"""You are a helpful AI research assistant. Use the following relevant research paper excerpts to answer the user's question. 
If you cannot answer the question based on the provided excerpts, say so.

{context}

User question: {query}

Please provide a clear and concise answer based on the above research excerpts. Include specific references to support your answer."""

            logger.info(f"Generated prompt with {len(sorted_chunks)} chunks of context")
            return prompt
            
        except Exception as e:
            logger.error(f"Error in generate_prompt: {str(e)}", exc_info=True)
            return f"Error generating prompt: {str(e)}"

    def query_llama(self, prompt: str) -> str:
        """
        Query Llama 3.2 through Ollama API.
        
        :param prompt: Formatted prompt
        :return: Llama's response
        """
        try:
            logger.info("Sending request to Ollama API...")
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "temperature": self.temperature,
                "stream": False
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30  # Add timeout
            )
            response.raise_for_status()
            
            result = response.json()["response"]
            logger.info("Successfully received response from Ollama")
            return result
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Error querying Llama: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg

    def answer_question(self, query: str) -> str:
        """
        Main method to answer a question using the RAG pipeline.
        
        :param query: User's question
        :return: Generated answer
        """
        try:
            logger.info(f"Processing question: {query}")
            
            # 1. Retrieve relevant chunks
            relevant_chunks = self.retrieve_relevant_chunks(query)
            
            if not relevant_chunks:
                return "No relevant information found in the research papers to answer your question."
            
            # 2. Generate prompt with context
            prompt = self.generate_prompt(query, relevant_chunks)
            
            # 3. Get response from Llama
            answer = self.query_llama(prompt)
            
            logger.info("Successfully generated answer")
            return answer
            
        except Exception as e:
            error_msg = f"Error in RAG pipeline: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg

def main():
    # Initialize RAG pipeline
    rag = RAGPipeline()
    
    print("\nWelcome to the Research Paper Q&A System!")
    print("Enter your questions below. Type 'exit' to quit.\n")
    
    while True:
        try:
            # Get user input
            query = input("\nYour question (or 'exit' to quit): ").strip()
            
            # Check if user wants to exit
            if query.lower() in ['exit', 'quit']:
                print("\nThank you for using the Research Paper Q&A System!")
                break
            
            # Skip empty queries
            if not query:
                print("Please enter a valid question.")
                continue
            
            print("\nGenerating answer...")
            answer = rag.answer_question(query)
            print(f"\nAnswer: {answer}")
            
        except KeyboardInterrupt:
            print("\n\nExiting the program...")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            logger.error("Error in main", exc_info=True)

if __name__ == "__main__":
    main() 
import logging
from typing import List, Dict
import os

from data_pipeline.arxiv_api import ArxivPaperFetcher
from data_pipeline.extract.pdf_extractor import PDFExtractor
from data_pipeline.extract.chunker import TextChunker
from data_pipeline.extract.text_preprocessor import TextPreprocessor
from data_pipeline.generate_embeddings import E5Embedder
from database.faiss_index import FaissIndex
from database.sqlite_db import DatabaseManager
from utils.file_operations import load_config

# class for paper pdf -> embedding -> store
class PDFPipeline:
    def __init__(self, batch_size=5):
        self.config = load_config()
        self.batch_size = batch_size
        self.db_path = self.config["database"]["arxiv_db_path"]
        self.root_path = self.config["storage"]["root_path"]
        self.paper_path = self.config["storage"]["save_path"]
        self.paper_abs_path = os.path.join(self.root_path, self.paper_path)
        self.faiss_index = FaissIndex()
        self.embedder = E5Embedder()
        self.arxiv_fetcher = ArxivPaperFetcher()
        
        # Ensure save path exists
        os.makedirs(self.root_path, exist_ok=True)
        os.makedirs(self.paper_abs_path, exist_ok=True)
    
    def _get_unprocessed_papers(self) -> List[Dict]:
        """Get papers that haven't had their text extracted yet or have missing PDFs."""
        with DatabaseManager(self.db_path) as cursor:
            cursor.execute("""
                SELECT paper_id, pdf_path, pdf_url 
                FROM papers 
                WHERE text_extracted = 0 OR pdf_path IS NULL
                LIMIT ?
            """, (self.batch_size,))
            
            # Get column names from cursor description
            columns = [description[0] for description in cursor.description]
            
            # Convert results to list of dictionaries
            results = cursor.fetchall()
            return [dict(zip(columns, row)) for row in results]
    
    def _ensure_pdf_exists(self, paper: Dict) -> str:
        """Ensures PDF exists, downloads if missing, and updates database."""
        if not paper.get("pdf_path"):
            if not paper.get("pdf_url"):
                raise ValueError(f"No PDF URL available for paper {paper['paper_id']}")
            
            logging.info(f"Attempting to download PDF for paper {paper['paper_id']}")
            
            # Use a single database connection for both operations
            with DatabaseManager(self.db_path) as cursor:
                # First check if the PDF already exists in the database
                cursor.execute("SELECT pdf_path FROM papers WHERE paper_id = ?", (paper["paper_id"],))
                result = cursor.fetchone()
                
                if result and result[0]:
                    logging.info(f"PDF path already exists in database for paper {paper['paper_id']}: {result[0]}")
                    return result[0]
                
                # If not, download the PDF
                self.arxiv_fetcher.download_arxiv_pdf(paper["paper_id"], paper["pdf_url"], cursor)
                
                # Get the updated pdf_path
                cursor.execute("SELECT pdf_path FROM papers WHERE paper_id = ?", (paper["paper_id"],))
                result = cursor.fetchone()
                logging.info(f"Database query result for paper {paper['paper_id']}: {result}")
                
                if not result:
                    logging.error(f"No database entry found for paper {paper['paper_id']}")
                    raise ValueError(f"No database entry found for paper {paper['paper_id']}")
                
                if not result[0]:
                    logging.error(f"PDF path is NULL for paper {paper['paper_id']}")
                    raise ValueError(f"PDF path is NULL for paper {paper['paper_id']}")
                
                return result[0]
        
        return paper["pdf_path"]
    
    def _process_paper(self, paper: Dict):
        """
        Process a single paper: extract text, chunk, generate embeddings, and store.
        Currently, storing embeddings in FAISS and paper and chunk data in SQLite.
        """
        try:
            # Ensure PDF exists and get its path
            pdf_path = self._ensure_pdf_exists(paper)
            
            # Convert relative path to absolute path
            absolute_pdf_path = os.path.join(self.paper_abs_path, pdf_path)
            logging.info(f"Processing PDF at absolute path: {absolute_pdf_path}")
            
            # Extract text from PDF
            extractor = PDFExtractor(absolute_pdf_path)
            extracted_data = extractor.process_pdf()
            
            # Combine all page text
            full_text = " ".join(extracted_data["text_data"].values())
            
            # Preprocess text
            preprocessor = TextPreprocessor(full_text, mode="passage")
            cleaned_text = preprocessor.clean_text()
            
            # Chunk text
            chunker = TextChunker(cleaned_text)
            chunks = chunker.chunk_text()
            
            logging.info(f"Generated {len(chunks)} chunks for paper {paper['paper_id']}")
            
            # Generate embeddings and store chunks
            # Pass chunks as a list to generate_embeddings to ensure consistent shapes
            self.faiss_index.insert_chunks_into_db(chunks, paper["paper_id"], 
                                                 lambda texts: self.embedder.generate_embeddings(texts, mode="passage"))
            
            # Update paper status
            with DatabaseManager(self.db_path) as cursor:
                cursor.execute("""
                    UPDATE papers 
                    SET text_extracted = 1 
                    WHERE paper_id = ?
                """, (paper["paper_id"],))
            
            logging.info(f"Successfully processed paper {paper['paper_id']}")
            
        except Exception as e:
            logging.error(f"Error processing paper {paper['paper_id']}: {e}")
            raise
    
    def process_pdfs(self):
        """Process unprocessed PDFs in batches."""
        try:
            # Get unprocessed papers
            papers = self._get_unprocessed_papers()
            
            if not papers:
                logging.info("No unprocessed papers found")
                return
            
            logging.info(f"Processing {len(papers)} papers")
            
            # Process each paper
            for paper in papers:
                self._process_paper(paper)
            
            logging.info("Batch processing complete")
            
        except Exception as e:
            logging.error(f"Error in process_pdfs: {e}")
            raise

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Initialize ArXiv fetcher and query for FAISS papers
    # arxiv_fetcher = ArxivPaperFetcher(query="Facebook AI Semantic Search (FAISS)")
    # papers = arxiv_fetcher.fetch_arxiv_paper_data(
    #     max_results=6
    # )
    # arxiv_fetcher.display_papers(papers)
    # arxiv_fetcher.store_papers_in_db(papers)
    
    # Initialize and run pipeline
    pipeline = PDFPipeline(batch_size=1)
    pipeline.process_pdfs()

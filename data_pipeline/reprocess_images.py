import os
import logging
import shutil
from typing import List, Optional
import time
from database.sqlite_db import DatabaseManager
from utils.file_operations import load_config
from extract.pdf_extractor import PDFExtractor

class PDFReprocessor:
    def __init__(self, batch_size=5):
        self.config = load_config()
        self.save_path = self.config["storage"]["save_path"]
        self.db_path = self.config["database"]["arxiv_db_path"]
        self.pdf_dir = os.path.join(self.save_path, "arxiv")
        self.image_dir = os.path.join(self.save_path, "arxiv/images")
        self.batch_size = batch_size
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def get_stored_pdfs(self) -> List[str]:
        """Get list of all stored PDFs from database."""
        with DatabaseManager(self.db_path) as cursor:
            cursor.execute("""
                SELECT paper_id, pdf_path 
                FROM papers 
                WHERE downloaded = 1
            """)
            return cursor.fetchall()

    def backup_image_dir(self) -> Optional[str]:
        """Create backup of existing images directory."""
        if not os.path.exists(self.image_dir):
            return None
            
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_dir = f"{self.image_dir}_backup_{timestamp}"
        
        try:
            shutil.copytree(self.image_dir, backup_dir)
            self.logger.info(f"Created backup of images at: {backup_dir}")
            return backup_dir
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")
            raise

    def clean_image_dir(self):
        """Remove all existing images to start fresh."""
        if os.path.exists(self.image_dir):
            shutil.rmtree(self.image_dir)
        os.makedirs(self.image_dir, exist_ok=True)

    def update_image_metadata(self, paper_id: str, image_data: List[dict]):
        """Update image metadata in database."""
        with DatabaseManager(self.db_path) as cursor:
            # First, remove old image records
            cursor.execute("""
                DELETE FROM images 
                WHERE paper_id = ?
            """, (paper_id,))
            
            # Insert new image records
            for img in image_data:
                cursor.execute("""
                    INSERT INTO images (
                        paper_id, page_id, page_num, image_path,
                        image_hash, size_bytes
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    paper_id,
                    img["page_id"],
                    img["page_num"],
                    img["image_path"],
                    img["image_hash"],
                    img["size_bytes"]
                ))

    def reprocess_pdfs(self):
        """Reprocess all stored PDFs with new image handling."""
        try:
            # Get list of PDFs to process
            stored_pdfs = self.get_stored_pdfs()
            total_pdfs = len(stored_pdfs)
            self.logger.info(f"Found {total_pdfs} PDFs to reprocess")

            # Backup existing images
            backup_dir = self.backup_image_dir()
            if backup_dir:
                self.logger.info(f"Backed up existing images to {backup_dir}")

            # Clean and recreate images directory
            self.clean_image_dir()
            self.logger.info("Cleaned images directory")

            # Process each PDF
            for idx, (paper_id, pdf_path) in enumerate(stored_pdfs, 1):
                try:
                    full_pdf_path = os.path.join(self.save_path, pdf_path)
                    if not os.path.exists(full_pdf_path):
                        self.logger.warning(f"PDF not found: {full_pdf_path}")
                        continue

                    self.logger.info(f"Processing PDF {idx}/{total_pdfs}: {paper_id}")
                    
                    # Extract images with new handling
                    extractor = PDFExtractor(
                        pdf_path=full_pdf_path,
                        output_folder=self.image_dir,
                        batch_size=self.batch_size
                    )
                    result = extractor.process_pdf()
                    
                    # Update database with new image metadata
                    self.update_image_metadata(paper_id, result["image_data"])
                    
                    self.logger.info(
                        f"Processed {paper_id}: "
                        f"{len(result['image_data'])} images, "
                        f"{result['stats']['total_size_mb']:.2f}MB total"
                    )

                except Exception as e:
                    self.logger.error(f"Error processing PDF {paper_id}: {e}")
                    continue

            self.logger.info("✅ Completed reprocessing all PDFs")

        except Exception as e:
            self.logger.error(f"Fatal error during reprocessing: {e}")
            if backup_dir:
                self.logger.info(f"Images backup available at: {backup_dir}")
            raise

def main():
    reprocessor = PDFReprocessor(batch_size=5)
    reprocessor.reprocess_pdfs()

if __name__ == "__main__":
    main() 
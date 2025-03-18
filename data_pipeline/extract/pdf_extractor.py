import pymupdf
import os
import json
import logging
from PIL import Image
import imagehash
import io
from pathlib import Path
from typing import Dict, List, Optional, Set
import time

class PDFExtractor:
    def __init__(self, pdf_path, output_folder="papers/arxiv/images", batch_size=5):
        self.pdf_path = pdf_path
        self.output_folder = output_folder
        self.batch_size = batch_size  # Process images in small batches
        self.hash_cache: Dict[str, Set[str]] = {}  # Cache for arxiv_id -> image hashes
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def compute_image_hash(self, img_bytes: bytes, img_ext: str) -> Optional[str]:
        """Compute perceptual hash of image using average hash algorithm."""
        try:
            # Convert bytes to PIL Image
            img = Image.open(io.BytesIO(img_bytes))
            # Compute average hash (fast and memory-efficient)
            hash_value = str(imagehash.average_hash(img, hash_size=8))
            return hash_value
        except Exception as e:
            self.logger.error(f"Failed to hash image: {e}")
            return None

    def is_duplicate_image(self, img_hash: str, arxiv_id: str) -> bool:
        """Check if image hash already exists for this paper."""
        if arxiv_id not in self.hash_cache:
            self.hash_cache[arxiv_id] = set()
        if img_hash in self.hash_cache[arxiv_id]:
            return True
        self.hash_cache[arxiv_id].add(img_hash)
        return False

    def extract_text(self):
        """Extracts text from the PDF and returns a structured dictionary."""
        doc = pymupdf.open(self.pdf_path)
        text_data = {}

        for page_num in range(len(doc)):
            page_text = doc[page_num].get_text("text")
            if page_text.strip():
                text_data[page_num] = page_text.strip()

        return text_data

    def process_image_batch(self, images: List[tuple], arxiv_id: str, page_id: str, page_num: int) -> List[dict]:
        """Process a batch of images efficiently."""
        batch_metadata = []
        
        for img_index, (xref, page) in enumerate(images):
            try:
                # Extract image data
                img_data = page.parent.extract_image(xref)
                img_bytes = img_data["image"]
                img_ext = img_data["ext"]

                # Compute hash before saving to disk
                img_hash = self.compute_image_hash(img_bytes, img_ext)
                if not img_hash:
                    continue

                # Skip if duplicate
                if self.is_duplicate_image(img_hash, arxiv_id):
                    self.logger.info(f"Skipping duplicate image on page {page_num + 1}")
                    continue

                # Format: {arxiv_id}_p{page_num}_i{img_index}.{ext}
                img_filename = f"{page_id}_i{img_index + 1}.{img_ext}"
                img_path = os.path.join(self.output_folder, img_filename)

                # Save image to disk
                with open(img_path, "wb") as f:
                    f.write(img_bytes)

                # Store metadata
                batch_metadata.append({
                    "page_id": page_id,
                    "page_num": page_num + 1,
                    "img_index": img_index + 1,
                    "image_path": img_path,
                    "arxiv_id": arxiv_id,
                    "image_hash": img_hash
                })

            except Exception as e:
                self.logger.error(f"Error processing image {img_index} on page {page_num + 1}: {e}")
                continue

        return batch_metadata

    def extract_images(self):
        """Extracts images with batched processing and deduplication."""
        doc = pymupdf.open(self.pdf_path)
        image_metadata = []

        # Extract arxiv_id from pdf_path (e.g., "2501.12948v1" from path)
        arxiv_id = self.pdf_path.split("/")[-1].split(".")[0]
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            # Create a unique page_id combining arxiv_id and page_num
            page_id = f"{arxiv_id}_p{page_num + 1}"
            
            # Collect all images from the page
            page_images = [(img[0], page) for img in page.get_images(full=True)]
            
            # Process images in batches
            for i in range(0, len(page_images), self.batch_size):
                batch = page_images[i:i + self.batch_size]
                batch_metadata = self.process_image_batch(batch, arxiv_id, page_id, page_num)
                image_metadata.extend(batch_metadata)
                
                # Small delay between batches to prevent thermal throttling
                if i + self.batch_size < len(page_images):
                    time.sleep(0.1)

        return image_metadata

    def process_pdf(self):
        """Runs both text and image extraction, returns structured metadata."""
        try:
            self.logger.info(f"Processing PDF: {self.pdf_path}")
            text_data = self.extract_text()
            image_data = self.extract_images()

            extracted_data = {
                "pdf_path": self.pdf_path,
                "text_data": text_data,
                "image_data": image_data,
            }
            
            self.logger.info(f"Processed {len(image_data)} unique images")
            return extracted_data
            
        except Exception as e:
            self.logger.error(f"Error processing PDF: {e}")
            raise

# Example usage
if __name__ == "__main__":
    pdf_path = "/Users/blake/Documents/Projects/summarize_research_papers/papers/arxiv/2501.12948v1.pdf"
    extractor = PDFExtractor(pdf_path)
    extracted_data = extractor.process_pdf()
    print(f"Extracted {len(extracted_data['image_data'])} unique images")

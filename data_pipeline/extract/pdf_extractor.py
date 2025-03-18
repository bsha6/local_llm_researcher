import pymupdf
import os
import logging
from PIL import Image
import imagehash
import io
from typing import Dict, List, Optional, Set, Generator
import time
from contextlib import contextmanager
import gc

class PDFExtractor:
    def __init__(self, pdf_path, output_folder="papers/arxiv/images", batch_size=5):
        self.pdf_path = pdf_path
        self.output_folder = output_folder
        self.batch_size = batch_size
        self.hash_cache: Dict[str, Set[str]] = {}
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    @contextmanager
    def open_pdf(self):
        """Context manager for safely handling PDF documents."""
        doc = None
        try:
            doc = pymupdf.open(self.pdf_path)
            yield doc
        finally:
            if doc:
                doc.close()
                gc.collect()  # Help clean up memory

    def compute_image_hash(self, img_bytes: bytes, img_ext: str) -> Optional[str]:
        """Compute perceptual hash of image using average hash algorithm."""
        try:
            with Image.open(io.BytesIO(img_bytes)) as img:
                # Resize image if it's too large (helps with memory)
                if img.size[0] > 2000 or img.size[1] > 2000:
                    img.thumbnail((2000, 2000), Image.Resampling.LANCZOS)
                # Compute average hash (fast and memory-efficient)
                hash_value = str(imagehash.average_hash(img, hash_size=8))
                return hash_value
        except Exception as e:
            self.logger.error(f"Failed to hash image: {e}")
            return None
        finally:
            gc.collect()  # Clean up PIL image objects

    def is_duplicate_image(self, img_hash: str, arxiv_id: str) -> bool:
        """Check if image hash already exists for this paper."""
        if arxiv_id not in self.hash_cache:
            self.hash_cache[arxiv_id] = set()
        if img_hash in self.hash_cache[arxiv_id]:
            return True
        self.hash_cache[arxiv_id].add(img_hash)
        return False

    def extract_text(self) -> Dict[int, str]:
        """Extracts text from the PDF and returns a structured dictionary."""
        text_data = {}
        with self.open_pdf() as doc:
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text("text")
                if page_text.strip():
                    text_data[page_num] = page_text.strip()
                # Clear page object to free memory
                page = None
                if page_num % 10 == 0:  # GC every 10 pages
                    gc.collect()
        return text_data

    def get_page_images(self, page) -> Generator[tuple, None, None]:
        """Generator for getting images from a page to reduce memory usage."""
        for img in page.get_images(full=True):
            yield img[0], page

    def process_image_batch(self, images: List[tuple], arxiv_id: str, page_id: str, page_num: int) -> List[dict]:
        """Process a batch of images efficiently."""
        batch_metadata = []
        page_duplicates = {}  # Track duplicates per page
        
        for img_index, (xref, page) in enumerate(images):
            try:
                # Extract image data
                img_data = page.parent.extract_image(xref)
                img_bytes = img_data["image"]
                img_ext = img_data["ext"]
                img_size = len(img_bytes)

                # Skip very large images
                if img_size > 10 * 1024 * 1024:  # Skip images larger than 10MB
                    self.logger.warning(f"Skipping large image ({img_size/1024/1024:.1f}MB) on page {page_num + 1}")
                    continue

                # Get image dimensions and additional metadata
                with Image.open(io.BytesIO(img_bytes)) as img:
                    width, height = img.size
                    # Skip tiny images (likely icons or artifacts)
                    if width < 20 or height < 20:
                        self.logger.debug(f"Skipping tiny image ({width}x{height}) on page {page_num + 1}")
                        continue
                    # Compute hash before saving to disk
                    img_hash = str(imagehash.average_hash(img, hash_size=8))

                # Track duplicates with more detail
                if self.is_duplicate_image(img_hash, arxiv_id):
                    if img_hash not in page_duplicates:
                        page_duplicates[img_hash] = {
                            "count": 1,
                            "size": f"{width}x{height}",
                            "bytes": img_size
                        }
                    else:
                        page_duplicates[img_hash]["count"] += 1
                    
                    self.logger.debug(
                        f"Duplicate on page {page_num + 1}: "
                        f"size={width}x{height}, "
                        f"bytes={img_size/1024:.1f}KB, "
                        f"occurrence={page_duplicates[img_hash]['count']}"
                    )
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
                    "image_hash": img_hash,
                    "size_bytes": img_size,
                    "width": width,
                    "height": height
                })

                # Clear references to help GC
                img_bytes = None
                img_data = None

            except Exception as e:
                self.logger.error(f"Error processing image {img_index} on page {page_num + 1}: {e}")
                continue

            # Periodic GC
            if img_index % 5 == 0:
                gc.collect()

        # Log duplicate statistics for this page
        if page_duplicates:
            total_dupes = sum(d["count"] for d in page_duplicates.values())
            self.logger.info(
                f"Page {page_num + 1} summary: "
                f"{len(page_duplicates)} unique duplicates, "
                f"{total_dupes} total duplicates"
            )
            for img_hash, stats in page_duplicates.items():
                self.logger.debug(
                    f"  Duplicate hash={img_hash[:8]}: "
                    f"found {stats['count']} times, "
                    f"size={stats['size']}, "
                    f"bytes={stats['bytes']/1024:.1f}KB"
                )

        return batch_metadata

    def extract_images(self) -> List[dict]:
        """Extracts images with batched processing and deduplication."""
        image_metadata = []
        arxiv_id = self.pdf_path.split("/")[-1].split(".pdf")[0]
        # TODO: ensure duplicate images are being handled correctly. Seeing lots of duplicates in logs.

        with self.open_pdf() as doc:
            total_pages = len(doc)
            for page_num in range(total_pages):
                page = doc[page_num]
                page_id = f"{arxiv_id}_p{page_num + 1}"
                
                # Use generator for memory efficiency
                page_images = []
                for img in self.get_page_images(page):
                    page_images.append(img)
                    
                    # Process in smaller batches if too many images on one page
                    if len(page_images) >= self.batch_size:
                        batch_metadata = self.process_image_batch(page_images, arxiv_id, page_id, page_num)
                        image_metadata.extend(batch_metadata)
                        page_images = []  # Clear processed images
                        time.sleep(0.05)  # Shorter sleep between same-page batches
                
                # Process remaining images
                if page_images:
                    batch_metadata = self.process_image_batch(page_images, arxiv_id, page_id, page_num)
                    image_metadata.extend(batch_metadata)
                
                # Progress logging
                if (page_num + 1) % 5 == 0:
                    self.logger.info(f"Processed {page_num + 1}/{total_pages} pages")
                    gc.collect()  # Regular GC
                
                # Longer delay between pages if processing many images
                if len(page_images) > 10:
                    time.sleep(0.1)

        return image_metadata

    def process_pdf(self) -> Dict:
        """Runs both text and image extraction, returns structured metadata."""
        try:
            self.logger.info(f"Processing PDF: {self.pdf_path}")
            
            # Process text and images
            text_data = self.extract_text()
            image_data = self.extract_images()

            # Calculate statistics
            total_size = sum(img["size_bytes"] for img in image_data)
            
            extracted_data = {
                "pdf_path": self.pdf_path,
                "text_data": text_data,
                "image_data": image_data,
                "stats": {
                    "total_images": len(image_data),
                    "total_size_mb": total_size / (1024 * 1024),
                    "avg_size_mb": total_size / (1024 * 1024 * len(image_data)) if image_data else 0
                }
            }
            
            self.logger.info(f"Processed {len(image_data)} unique images, "
                           f"Total size: {extracted_data['stats']['total_size_mb']:.2f}MB, "
                           f"Avg size: {extracted_data['stats']['avg_size_mb']:.2f}MB")
            return extracted_data
            
        except Exception as e:
            self.logger.error(f"Error processing PDF: {e}")
            raise
        finally:
            # Clear memory
            self.hash_cache.clear()
            gc.collect()

# Example usage
if __name__ == "__main__":
    pdf_path = "/Users/blake/Documents/Projects/summarize_research_papers/papers/arxiv/2501.12948v1.pdf"
    extractor = PDFExtractor(pdf_path, batch_size=5)
    extracted_data = extractor.process_pdf()
    print(f"Extracted {extracted_data['stats']['total_images']} unique images")

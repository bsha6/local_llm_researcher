import pymupdf
import os
import json

class PDFExtractor:
    def __init__(self, pdf_path, output_folder="papers/arxiv/images"):
        self.pdf_path = pdf_path
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

    def extract_text(self):
        """Extracts text from the PDF and returns a structured dictionary."""
        doc = pymupdf.open(self.pdf_path)
        text_data = {}

        for page_num in range(len(doc)):
            page_text = doc[page_num].get_text("text")
            if page_text.strip():
                text_data[page_num] = page_text.strip()

        return text_data

    def extract_images(self):
        """Extracts images and stores them by page number."""
        doc = pymupdf.open(self.pdf_path)
        image_metadata = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                img_data = doc.extract_image(xref)
                img_bytes = img_data["image"]
                img_ext = img_data["ext"]

                img_filename = f"page_{page_num + 1}_img_{img_index}.{img_ext}"
                img_path = os.path.join(self.output_folder, img_filename)

                with open(img_path, "wb") as f:
                    f.write(img_bytes)

                image_metadata.append({"page": page_num, "image_path": img_path})

        return image_metadata

    def process_pdf(self):
        """Runs both text and image extraction, returns structured metadata."""
        # TODO: implement multi-threading (use 6/8 CPU cores)
        text_data = self.extract_text()
        image_data = self.extract_images()

        extracted_data = {
            "pdf_path": self.pdf_path,
            "text_data": text_data,
            "image_data": image_data,
        }

        # IF this process becomes blocking/slow: Save extracted data to sqlite db?
        # json_path = os.path.join(self.output_folder, "extracted_data.json")
        # with open(json_path, "w") as f:
        #     json.dump(extracted_data, f, indent=4)
        
        return extracted_data

# Example usage
if __name__ == "__main__":
    pdf_path = "/Users/blake/Documents/Projects/summarize_research_papers/papers/arxiv/2501.12948v1.pdf"
    # extractor = PDFExtractor(pdf_path)
    # extractor.process_pdf()
    print("Extraction complete.")

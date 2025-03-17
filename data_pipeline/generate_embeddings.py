import torch
from transformers import AutoTokenizer, AutoModel

class E5Embedder:
    def __init__(self, model_name="intfloat/multilingual-e5-small", device=None):
        """
        Initializes the E5-Small model and tokenizer.
        :param model_name: Model name from Hugging Face.
        :param device: "cpu" or "cuda" (auto-detected if None).
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()  # Load model in eval mode

    def preprocess_text(self, texts, mode="passage"):
        """
        Prepares text with the required prefix: "query: " or "passage: ".
        :param texts: List of input texts.
        :param mode: "query" for search queries, "passage" for documents.
        """
        assert mode in ["query", "passage"], "Mode must be 'query' or 'passage'."
        return [f"{mode}: {text.strip()}" for text in texts]

    def generate_embeddings(self, texts, mode="passage"):
        """
        Generates embeddings for a list of texts.
        :param texts: List of input texts.
        :param mode: "query" or "passage" (default: passage).
        :return: NumPy array of embeddings.
        """
        texts = self.preprocess_text(texts, mode)
        encoded_inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        encoded_inputs = {k: v.to(self.device) for k, v in encoded_inputs.items()}  # Move to device

        with torch.no_grad():  # Disable gradient calculation for efficiency
            model_output = self.model(**encoded_inputs)

        # Use the [CLS] token embedding (first token) as the sentence representation
        embeddings = model_output.last_hidden_state[:, 0, :].cpu().numpy()
        return embeddings
    
    def generate_image_embeddings(self, image_path):
        # TODO: go through all images, convert to text, and generate embedding. 
        # Could use Blip-small to convert from image to text
        # Delete image after generating embedding/some way of making sure this only happens once
        pass

# Example Usage:
if __name__ == "__main__":
    embedder = E5Embedder()

    # Example sentences
    texts = [
        "AI is transforming research in various domains.",
        "Deep learning has improved natural language processing."
    ]

    embeddings = embedder.generate_embeddings(texts, mode="passage")
    
    # Print shape and first embedding
    print("Embedding shape:", embeddings.shape)
    print("First embedding vector:", embeddings[0])

import re
from itertools import accumulate
from transformers import AutoTokenizer

class TextChunker:
    def __init__(self, text, model="intfloat/multilingual-e5-small", max_tokens=512, overlap=50):
        """
        :param text: The cleaned text to be chunked.
        :param model: Tokenizer model name (Hugging Face).
        :param max_tokens: Max tokens per chunk.
        :param overlap: Overlapping tokens for context preservation.
        """
        self.text = text
        self.tokenizer = AutoTokenizer.from_pretrained(model)  # Use HF tokenizer
        self.max_tokens = max_tokens
        self.overlap = overlap

    def split_into_sentences(self):
        """Splits text into sentences while keeping structure."""
        sentences = re.split(r'(?<=[.!?])\s+', self.text)  # Split on sentence boundaries
        return [s.strip() for s in sentences if s.strip()]

    def tokenize_sentences(self, sentences):
        """Tokenizes sentences and computes cumulative token counts."""
        token_counts = [len(self.tokenizer.encode(sent, add_special_tokens=False)) for sent in sentences]
        cumulative_tokens = list(accumulate(token_counts))
        return sentences, token_counts, cumulative_tokens

    def chunk_sentences(self, sentences, token_counts, cumulative_tokens):
        """Chunks text while ensuring sentence boundaries and preserving context."""
        chunks = []
        current_chunk = []
        current_length = 0
        start_idx = 0

        for i, (sentence, token_count) in enumerate(zip(sentences, token_counts)):
            if current_length + token_count > self.max_tokens:
                chunks.append(" ".join(current_chunk))  # Save chunk
                
                # Use sliding window
                start_idx = max(0, i - (self.overlap // token_count))  
                current_chunk = sentences[start_idx:i]  
                current_length = sum(token_counts[start_idx:i])

            current_chunk.append(sentence)
            current_length += token_count

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def chunk_text(self):
        """Runs the full chunking pipeline."""
        sentences = self.split_into_sentences()
        sentences, token_counts, cumulative_tokens = self.tokenize_sentences(sentences)
        return self.chunk_sentences(sentences, token_counts, cumulative_tokens)

# Example usage:
if __name__ == "__main__":
    sample_text = """
    2.1 Overview. Deep learning models have evolved rapidly. The introduction of large-scale reinforcement learning models has significantly improved performance.
    However, challenges remain in scaling. Our approach builds upon existing methods.
    2.2 Reinforcement Learning. We propose a novel training methodology.
    """

    chunker = TextChunker(sample_text)
    chunks = chunker.chunk_text()
    
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:\n{chunk}\n{'='*50}")

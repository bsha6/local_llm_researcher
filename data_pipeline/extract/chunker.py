import re
from itertools import accumulate
from transformers import AutoTokenizer
import logging

class TextChunker:
    def __init__(self, text, model="intfloat/multilingual-e5-small", max_tokens=450, overlap=50):
        """
        :param text: The cleaned text to be chunked.
        :param model: Tokenizer model name (Hugging Face).
        :param max_tokens: Max tokens per chunk (default lowered to 450 to account for special tokens).
        :param overlap: Overlapping tokens for context preservation.
        """
        self.text = text
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.max_tokens = max_tokens
        self.overlap = overlap
        self.logger = logging.getLogger(__name__)

    def _get_token_count(self, text: str, add_special_tokens: bool = True) -> int:
        """Helper method to get token count for a piece of text.
        
        Args:
            text: Text to tokenize
            add_special_tokens: Whether to account for special tokens
        Returns:
            Token count (excluding special tokens if add_special_tokens=True)
        """
        count = len(self.tokenizer.encode(text, add_special_tokens=add_special_tokens))
        return count - 2 if add_special_tokens else count

    def _chunk_text_by_words(self, text: str) -> list:
        """Split a piece of text into chunks at word boundaries.
        
        Args:
            text: Text to split into chunks
        Returns:
            List of chunks, each guaranteed to be under max_tokens
        """
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            word_tokens = self._get_token_count(word)
            
            # If single word is too long, log warning and skip
            if word_tokens + 2 > self.max_tokens:
                self.logger.warning(f"Single word exceeds token limit, truncating: {word[:50]}...")
                continue
                
            if current_length + word_tokens + 2 > self.max_tokens:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = word_tokens
            else:
                current_chunk.append(word)
                current_length += word_tokens

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def split_into_sentences(self):
        """Splits text into sentences while keeping structure."""
        sentences = re.split(r'(?<=[.!?])\s+', self.text)
        return [s.strip() for s in sentences if s.strip()]

    def tokenize_sentences(self, sentences):
        """Tokenizes sentences and computes cumulative token counts."""
        token_counts = [self._get_token_count(sent) for sent in sentences]
        cumulative_tokens = list(accumulate(token_counts))
        return sentences, token_counts, cumulative_tokens

    def chunk_sentences(self, sentences, token_counts, cumulative_tokens):
        """Chunks text while ensuring sentence boundaries and token limits."""
        chunks = []
        current_chunk = []
        current_length = 0

        for i, (sentence, token_count) in enumerate(zip(sentences, token_counts)):
            # Handle sentences that are too long
            if token_count + 2 > self.max_tokens:
                # Add any accumulated sentences first
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Split long sentence into word-level chunks
                chunks.extend(self._chunk_text_by_words(sentence))
                continue

            # Handle normal sentences
            if current_length + token_count + 2 > self.max_tokens:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = token_count
            else:
                current_chunk.append(sentence)
                current_length += token_count

        # Add remaining sentences
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        # Final verification pass
        verified_chunks = []
        for chunk in chunks:
            tokens = self._get_token_count(chunk)
            if tokens + 2 > self.max_tokens:
                # If still too long, split at word level
                verified_chunks.extend(self._chunk_text_by_words(chunk))
            else:
                verified_chunks.append(chunk)

        # Final length check and logging
        for chunk in verified_chunks:
            total_tokens = self._get_token_count(chunk) + 2  # Add special tokens
            if total_tokens > self.max_tokens:
                self.logger.error(f"Chunk still exceeds token limit: {total_tokens} tokens")

        return verified_chunks

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
    I've made several significant changes to ensure no chunk exceeds the token limit:
    Added a check for individual sentences that are too long, splitting them into smaller pieces at the word level
    Removed the overlap logic which wasn't effectively controlling chunk size
    Added a more aggressive final verification step that will split any remaining overlength chunks at the word level
    Made the chunking process more linear and predictable
    Added multiple levels of verification to ensure chunks stay within limits
    The new logic:
    First checks if individual sentences are too long and splits them if needed
    Then builds chunks sentence by sentence, starting a new chunk whenever adding a sentence would exceed the limit
    Finally verifies all chunks and splits any that are still too long at the word level
    I'll help you investigate the chunking process. Let's first look at the chunker implementation to understand how the text is being split.
    Looking at the code, I can see that the issue is coming from the token length in the E5Embedder class. The error occurs because the text chunks being passed to the embedder are longer than the model's maximum sequence length of 512 tokens.
    Let's look at how the chunking is currently implemented to fix this issue:
    The TextChunker class in data_pipeline/extract/chunker.py is responsible for splitting the text into chunks
    The current implementation uses sentence boundaries and a token count check, but it seems the token counting logic might not be working as expected
    Here's what we need to fix:
    The chunk_sentences method in TextChunker needs to be more strict about enforcing the token limit
    We should account for the special tokens that are added by the tokenizer (like [CLS], [SEP])
    We should reduce the maximum tokens per chunk to leave room for these special tokens
    Let me propose an edit to fix this:
    """

    chunker = TextChunker(sample_text)
    chunks = chunker.chunk_text()
    
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:\n{chunk}\n{'='*50}")

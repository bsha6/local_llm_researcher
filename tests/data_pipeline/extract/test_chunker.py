import pytest

from data_pipeline.extract.chunker import TextChunker


@pytest.mark.usefixtures("setup_config_loader")
class TestTextChunker:
    """Test suite for the TextChunker class."""
    def test_init(self, mock_tokenizer, setup_config_loader):
        """Test the initialization of TextChunker."""
        chunker = TextChunker("Sample text")
        
        # Verify tokenizer was initialized with correct model
        mock_tokenizer.assert_called_once_with("intfloat/multilingual-e5-small")
        
        # Verify default parameters
        assert chunker.max_tokens == setup_config_loader['models']['e5_small']['max_tokens']
        assert chunker.overlap == 50
        assert chunker.text == "Sample text"

    def test_split_into_sentences(self, sample_text):
        """Test splitting text into sentences."""
        chunker = TextChunker(sample_text)
        sentences = chunker.split_into_sentences()
        
        # Check we have the expected number of sentences
        assert len(sentences) == 7
        
        # Check first and last sentences
        assert sentences[0].startswith("2.1 Overview")
        assert sentences[-1].startswith("We propose")

    def test_tokenize_sentences(self, sample_text, mock_tokenizer):
        """Test tokenization of sentences."""
        chunker = TextChunker(sample_text)
        sentences = chunker.split_into_sentences()
        sentences, token_counts, cumulative_tokens = chunker.tokenize_sentences(sentences)
        
        # Verify all lists have the same length
        assert len(sentences) == len(token_counts) == len(cumulative_tokens)
        
        # Verify cumulative tokens is correctly calculated
        assert cumulative_tokens[0] == token_counts[0]
        for i in range(1, len(cumulative_tokens)):
            assert cumulative_tokens[i] == cumulative_tokens[i-1] + token_counts[i]

    def test_chunk_sentences(self, sample_text, mock_tokenizer):
        """Test chunking sentences based on token limits."""
        # Create a chunker with a small max_tokens to force multiple chunks
        chunker = TextChunker(sample_text, max_tokens=20, overlap=5)
        sentences = chunker.split_into_sentences()
        sentences, token_counts, cumulative_tokens = chunker.tokenize_sentences(sentences)
        chunks = chunker.chunk_sentences(sentences, token_counts, cumulative_tokens)
        
        # Verify we get multiple chunks with our small token limit
        assert len(chunks) > 1
        
        # Verify each chunk is a non-empty string
        for chunk in chunks:
            assert isinstance(chunk, str)
            assert len(chunk) > 0

    def test_chunk_text_integration(self, sample_text, mock_tokenizer):
        """Integration test for the full chunking pipeline."""
        chunker = TextChunker(sample_text, max_tokens=100, overlap=10)
        chunks = chunker.chunk_text()
        
        # Verify we get chunks
        assert len(chunks) > 0
        
        # Verify all original sentences are represented in the chunks
        combined_chunks = " ".join(chunks)
        sentences = chunker.split_into_sentences()
        for sentence in sentences:
            assert sentence in combined_chunks

    def test_with_empty_text(self, mock_tokenizer):
        """Test behavior with empty text."""
        chunker = TextChunker("")
        chunks = chunker.chunk_text()
        assert len(chunks) == 0

    def test_with_single_sentence(self, mock_tokenizer):
        """Test behavior with a single sentence."""
        single_sentence = "This is just one sentence."
        chunker = TextChunker(single_sentence)
        chunks = chunker.chunk_text()
        assert len(chunks) == 1
        assert chunks[0] == single_sentence

    def test_custom_parameters(self, sample_text, mock_tokenizer):
        """Test with custom parameters."""
        custom_model = "bert-base-uncased"
        custom_max_tokens = 256
        custom_overlap = 25
        
        chunker = TextChunker(
            sample_text, 
            model=custom_model, 
            max_tokens=custom_max_tokens, 
            overlap=custom_overlap
        )
        
        assert chunker.max_tokens == custom_max_tokens
        assert chunker.overlap == custom_overlap
        mock_tokenizer.assert_called_once_with(custom_model)

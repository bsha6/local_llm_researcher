import pytest
import torch
import numpy as np
from transformers import PreTrainedTokenizerFast
from unittest.mock import MagicMock

from data_pipeline.generate_embeddings import E5Embedder

class TestE5Embedder:
    @pytest.fixture
    def embedder(self, mocker):
        """Fixture to initialize E5Embedder with mocks for CI testing."""
        # Mock the tokenizer
        mock_tokenizer = MagicMock(spec=PreTrainedTokenizerFast)
        mock_tokenizer.batch_encode_plus.return_value = {
            "input_ids": torch.tensor([[101, 2001, 102], [101, 2002, 102]]),  # Simulate two inputs
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]])
        }
        mock_tokenizer.return_tensors = "pt"
        mocker.patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer)

        # Mock the model
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model

        # Create a real NumPy array for the mock output
        test_array = np.random.randn(1, 384)

        # Create a proper mock output structure
        mock_output = MagicMock()
        mock_output.last_hidden_state = MagicMock()
        
        # Create a mock tensor that will handle the chain of operations
        mock_tensor = MagicMock()
        mock_tensor.cpu.return_value = mock_tensor
        mock_tensor.numpy.return_value = test_array
        
        # Set up the mock chain to return our test array
        # Handle the indexing operation [:, 0, :]
        mock_output.last_hidden_state.__getitem__.return_value = mock_tensor

        # Set up the model's forward method to return our mock output
        mock_model.forward = MagicMock(return_value=mock_output)
        mocker.patch("transformers.AutoModel.from_pretrained", return_value=mock_model)

        # Initialize and return embedder
        return E5Embedder("mock-model-path")
        

    def test_model_initialization(self, embedder):
        """Test that the model and tokenizer are correctly initialized."""
        assert embedder.tokenizer is not None, "Tokenizer should be initialized"
        assert embedder.model is not None, "Model should be initialized"
        assert embedder.device in ["cpu", "cuda"], "Device should be either 'cpu' or 'cuda'"

    def test_preprocess_text(self, embedder):
        """Test that text preprocessing correctly adds prefixes."""
        texts = ["This is a test sentence."]
        processed_passage = embedder.preprocess_text(texts, mode="passage")
        processed_query = embedder.preprocess_text(texts, mode="query")

        assert processed_passage == ["passage: This is a test sentence."]
        assert processed_query == ["query: This is a test sentence."]

    def test_invalid_mode(self, embedder):
        """Test that an invalid mode raises an assertion error."""
        with pytest.raises(AssertionError):
            embedder.preprocess_text(["Some text"], mode="invalid")

    def test_generate_embeddings(self, embedder):
        """Test that embeddings are generated with the expected shape."""
        dummy_tensor = torch.randn(1, 10, 768)
    
        # Create a dummy output object with a 'last_hidden_state' attribute.
        dummy_output = type("DummyOutput", (), {})()
        dummy_output.last_hidden_state = dummy_tensor

        # Patch the model so that when it's called, it returns our dummy output.
        embedder.model = MagicMock(return_value=dummy_output)

        texts = ["AI is revolutionizing research."]
        embeddings = embedder.generate_embeddings(texts, mode="passage")

        # Verify that the embeddings are a NumPy array.
        assert isinstance(embeddings, np.ndarray), "Embeddings should be a NumPy array"
        # Optionally, also check that the shape matches what you expect.
        # Since we're taking the first token ([CLS]) from each input, the expected shape is (batch_size, hidden_dim).
        assert embeddings.shape == (1, 768), f"Unexpected embedding shape: {embeddings.shape}"
        

    def test_generate_embeddings_multiple_inputs(self, embedder):
        """Test that embeddings are generated correctly for multiple inputs."""
         # Create a dummy tensor: 2 texts, sequence length of 10, and hidden dimension of 384.
        dummy_tensor = torch.randn(2, 10, 384)
        
        # Create a dummy output object with a 'last_hidden_state' attribute.
        dummy_output = type("DummyOutput", (), {})()
        dummy_output.last_hidden_state = dummy_tensor

        # Patch the embedder's model so that it returns our dummy output.
        embedder.model = MagicMock(return_value=dummy_output)
        
        texts = ["Text 1", "Text 2"]
        embeddings = embedder.generate_embeddings(texts, mode="passage")

        assert isinstance(embeddings, np.ndarray), "Embeddings should be a NumPy array"
        assert embeddings.shape == (2, 384), "Embedding shape should match (2, 384) for multiple inputs"


if __name__ == "__main__":
    pytest.main()

import pytest
import torch
import numpy as np
from transformers import PreTrainedTokenizerFast
from unittest.mock import MagicMock

from data_pipeline.generate_embeddings import E5Embedder  # Adjust import path based on your project structure

@pytest.fixture
def embedder(mocker):
    """Fixture to initialize E5Embedder with mocks for CI testing."""
    # Mock the tokenizer
    mock_tokenizer = MagicMock(spec=PreTrainedTokenizerFast)
    mock_tokenizer.return_value = mock_tokenizer
    mocker.patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer)
    
    # Mock the model
    mock_model = MagicMock()
    mock_model.eval.return_value = mock_model
    class MockOutput:
        def __init__(self):
            self.last_hidden_state = torch.randn(1, 512, 384)
    mock_model.return_value = MockOutput()
    mocker.patch("transformers.AutoModel.from_pretrained", return_value=mock_model)
    
    return E5Embedder(model_name="intfloat/multilingual-e5-small", device="cpu")

def test_model_initialization(embedder):
    """Test that the model and tokenizer are correctly initialized."""
    assert isinstance(embedder.tokenizer, PreTrainedTokenizerFast), "Tokenizer should be an instance of PreTrainedTokenizerFast"
    assert embedder.model is not None, "Model should be initialized"
    assert embedder.device in ["cpu", "cuda"], "Device should be either 'cpu' or 'cuda'"

def test_preprocess_text(embedder):
    """Test that text preprocessing correctly adds prefixes."""
    texts = ["This is a test sentence."]
    processed_passage = embedder.preprocess_text(texts, mode="passage")
    processed_query = embedder.preprocess_text(texts, mode="query")

    assert processed_passage == ["passage: This is a test sentence."]
    assert processed_query == ["query: This is a test sentence."]

def test_invalid_mode(embedder):
    """Test that an invalid mode raises an assertion error."""
    with pytest.raises(AssertionError):
        embedder.preprocess_text(["Some text"], mode="invalid")

@pytest.fixture
def mock_model(mocker):
    """Fixture to mock the Hugging Face model to speed up testing."""
    mock = mocker.patch("transformers.AutoModel.from_pretrained")  # Mock the model loading
    mock_instance = mock.return_value
    mock_instance.eval.return_value = mock_instance  # Mock eval mode to return self
    
    # Create a mock output with last_hidden_state
    class MockOutput:
        def __init__(self):
            self.last_hidden_state = torch.randn(2, 512, 384)
    
    # Make the mock model return the mock output when called
    mock_instance.return_value = MockOutput()
    
    return mock_instance

def test_generate_embeddings(embedder, mocker):
    """Test that embeddings are generated with the expected shape."""
    # Create a mock output with last_hidden_state
    class MockOutput:
        def __init__(self):
            self.last_hidden_state = torch.randn(1, 512, 384)
    
    # Mock the model to return our mock output
    mock_model = mocker.MagicMock()
    mock_model.return_value = MockOutput()
    mocker.patch.object(embedder, "model", mock_model)
    
    texts = ["AI is revolutionizing research."]
    embeddings = embedder.generate_embeddings(texts, mode="passage")

    assert isinstance(embeddings, np.ndarray), "Embeddings should be a NumPy array"
    assert embeddings.shape == (1, 384), "Embedding shape should match (1, 384)"

def test_generate_embeddings_multiple_inputs(embedder, mocker):
    """Test that embeddings are generated correctly for multiple inputs."""
    # Create a mock output with last_hidden_state
    class MockOutput:
        def __init__(self):
            self.last_hidden_state = torch.randn(2, 512, 384)
    
    # Mock the model to return our mock output
    mock_model = mocker.MagicMock()
    mock_model.return_value = MockOutput()
    mocker.patch.object(embedder, "model", mock_model)

    texts = ["Text 1", "Text 2"]
    embeddings = embedder.generate_embeddings(texts, mode="passage")

    assert embeddings.shape == (2, 384), "Embedding shape should match (2, 384) for multiple inputs"

if __name__ == "__main__":
    pytest.main()

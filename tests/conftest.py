import pytest
import numpy as np
import tempfile
import json
from unittest.mock import patch, MagicMock

from database.faiss_index import FaissIndex

"""
Configuration file for pytest.
This file is automatically recognized by pytest and can define fixtures,
hooks, and other test setup that should be available across multiple test files.
"""

@pytest.fixture
def mock_config():
    """Fixture that provides a mock configuration dictionary for testing."""
    return {
        "database": {
            "arxiv_db_path": ":memory:"
        },
        "arxiv": {
            "query": "test query"
        },
        "storage": {
            "save_path": "test_papers/",
        }
    }

@pytest.fixture
def temp_config_file(tmp_path):
    """Fixture to create a temporary config file."""
    temp_file = tmp_path / "test_config.yaml"  # Ensure this is a Path object
    config_data = {
        "arxiv": {"query": "test query"},
        "database": {"arxiv_db_path": ":memory:"},
        "storage": {"save_path": "test_papers/"}
    }
    
    with open(temp_file, "w") as f:
        json.dump(config_data, f)
    
    return temp_file  # Return a Path object, not a string

@pytest.fixture
def temp_directory():
    """Fixture that creates a temporary directory for testing file operations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

# ========== MOCKS FOR GHOSTSCRIPT (GS) ========== #

@pytest.fixture
def mock_gs_path():
    """Fixture that returns a mock path to the Ghostscript executable."""
    return "/mock/path/to/gs"

@pytest.fixture
def mock_subprocess_run():
    """Fixture that mocks subprocess.run"""
    with patch("subprocess.run") as mock_run:
        yield mock_run

@pytest.fixture
def mock_shutil_which():
    """Fixture that mocks shutil.which to return a path to gs"""
    with patch("shutil.which") as mock_which:
        mock_which.return_value = "/mocked/gs/path"
        yield mock_which

@pytest.fixture
def mock_os_path_exists():
    """Fixture that mocks os.path.exists with dynamic behavior"""
    with patch("os.path.exists") as mock_exists:
        yield mock_exists


'''FAISS'''

@pytest.fixture
def mock_faiss_index(mocker):
    """Fixture to mock FAISS index and avoid file I/O."""
    mock_index = MagicMock()
    mock_index.add = MagicMock()
    mock_index.search = MagicMock(return_value=(np.array([[0, 1, 2]]), np.array([[0.1, 0.2, 0.3]])))
    mock_index.ntotal = 0  # Mock number of stored vectors

    # Mock FAISS index types
    mocker.patch("faiss.IndexHNSWFlat", return_value=mock_index)
    mocker.patch("faiss.IndexFlatL2", return_value=mock_index)
    mocker.patch("faiss.read_index", return_value=mock_index)
    mocker.patch("faiss.write_index")  # Prevents actual file writing

    return mock_index

@pytest.fixture
def mock_db_manager(mocker):
    """Fixture to mock DatabaseManager for FAISS index tests."""
    mock_manager = MagicMock()
    mock_cursor = MagicMock()
    mock_manager.__enter__.return_value = mock_cursor
    mock_cursor.executemany = MagicMock()
    mock_cursor.execute = MagicMock()
    mock_cursor.fetchall = MagicMock(return_value=[])
    
    mocker.patch('database.faiss_index.DatabaseManager', return_value=mock_manager)
    
    return mock_manager

@pytest.fixture
def faiss_index(mock_faiss_index, mock_db_manager):
    """Fixture for an initialized FaissIndex instance."""
    index = FaissIndex()
    # Ensure _ensure_faiss_idx_column has been called
    return index

'''Chunking'''

@pytest.fixture
def sample_text():
    """Fixture providing sample text for testing."""
    return """
    2.1 Overview. Deep learning models have evolved rapidly. The introduction of large-scale reinforcement learning models has significantly improved performance.
    However, challenges remain in scaling. Our approach builds upon existing methods.
    2.2 Reinforcement Learning. We propose a novel training methodology.
    """


@pytest.fixture
def mock_tokenizer():
    """Fixture that mocks the HuggingFace tokenizer."""
    with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
        tokenizer_instance = MagicMock()
        # Configure the encode method to return a reasonable number of tokens
        tokenizer_instance.encode.side_effect = lambda sent, add_special_tokens: [0] * (len(sent.split()) + 2)
        mock_tokenizer.return_value = tokenizer_instance
        yield mock_tokenizer

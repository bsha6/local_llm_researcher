import pytest
from unittest.mock import patch
import os

# Mock config before importing any modules that might use it
@pytest.fixture(autouse=True)
def mock_config_globally():
    """Fixture to mock config loading globally before any imports."""
    mock_config = {
        "database": {
            "arxiv_db_path": ":memory:",
            "faiss_index_path": "test_index.idx"
        },
        "arxiv": {
            "query": "test query"
        },
        "storage": {
            "root_path": "/test/root",
            "save_path": "test_papers/"
        }
    }
    with patch("utils.file_operations.load_config", return_value=mock_config):
        yield mock_config

# Now we can safely import modules that depend on config
import numpy as np
import tempfile
import json
from unittest.mock import MagicMock

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
            "root_path": "/test/root",
            "save_path": "test_papers/",
        }
    }

@pytest.fixture
def temp_config_file(tmp_path):
    """Fixture to create a temporary config file."""
    temp_file = tmp_path / "test_config.yaml"  # Ensure this is a Path object
    config_data = {
        "database": {
            "arxiv_db_path": ":memory:"
        },
        "arxiv": {
            "query": "test query"
        },
        "storage": {
            "root_path": "/test/root",
            "save_path": "test_papers/",
        }
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

'''PDF Pipeline'''

@pytest.fixture
def mock_database():
    """Fixture to mock database operations."""
    mock_db = MagicMock()
    mock_cursor = MagicMock()
    mock_db.__enter__.return_value = mock_cursor
    mock_cursor.execute = MagicMock()
    mock_cursor.fetchall = MagicMock(return_value=[])
    mock_cursor.fetchone = MagicMock()
    mock_cursor.executemany = MagicMock()
    return mock_db, mock_cursor

@pytest.fixture
def mock_faiss():
    """Fixture to mock FAISS operations."""
    mock_faiss = MagicMock()
    mock_faiss.insert_chunks_into_db = MagicMock()
    mock_faiss.search = MagicMock(return_value=(np.array([[0, 1, 2]]), np.array([[0.1, 0.2, 0.3]])))
    return mock_faiss

@pytest.fixture
def mock_pdf_pipeline_dependencies(mocker, mock_database, mock_faiss):
    """Mock all external dependencies for PDFPipeline."""
    mock_db, mock_cursor = mock_database
    
    # Mock ArxivPaperFetcher
    mock_fetcher = MagicMock()
    mock_fetcher.download_arxiv_pdf = MagicMock()
    mocker.patch('main.ArxivPaperFetcher', return_value=mock_fetcher)
    
    # Mock PDFExtractor
    mock_extractor = MagicMock()
    mock_extractor.process_pdf = MagicMock(return_value={"text_data": {"1": "Sample text"}})
    mocker.patch('main.PDFExtractor', return_value=mock_extractor)
    
    # Mock TextPreprocessor
    mock_preprocessor = MagicMock()
    mock_preprocessor.clean_text = MagicMock(return_value="Cleaned sample text")
    mocker.patch('main.TextPreprocessor', return_value=mock_preprocessor)
    
    # Mock TextChunker
    mock_chunker = MagicMock()
    mock_chunker.chunk_text = MagicMock(return_value=["Chunk 1", "Chunk 2"])
    mocker.patch('main.TextChunker', return_value=mock_chunker)
    
    # Mock E5Embedder
    mock_embedder = MagicMock()
    mock_embedder.generate_embeddings = MagicMock(return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    mocker.patch('main.E5Embedder', return_value=mock_embedder)
    
    # Mock FaissIndex and DatabaseManager
    mocker.patch('main.FaissIndex', return_value=mock_faiss)
    mocker.patch('main.DatabaseManager', return_value=mock_db)
    
    return {
        'fetcher': mock_fetcher,
        'extractor': mock_extractor,
        'preprocessor': mock_preprocessor,
        'chunker': mock_chunker,
        'embedder': mock_embedder,
        'faiss': mock_faiss,
        'db': mock_db,
        'cursor': mock_cursor
    }

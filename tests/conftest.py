import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import tempfile
import json
import os
import logging

from utils.file_operations import ConfigLoader

# Set up logging
logger = logging.getLogger(__name__)

# Define a consistent mock config structure
MOCK_CONFIG = {
    "database": {
        "arxiv_db_path": ":memory:",
        "faiss_index_path": "test_index.idx"
    },
    "arxiv": {
        "query": "test query"
    },
    "storage": {
        "save_path": "test_papers/"
    },
    "models": {
        "e5_small": {
            "dimensions": 384
        }
    }
}


@pytest.fixture
def temp_index_path():
    """Fixture to create a temporary path for the FAISS index."""
    with tempfile.NamedTemporaryFile(suffix='.idx', delete=False) as tmp:
        yield tmp.name
        # Cleanup
        if os.path.exists(tmp.name):
            os.remove(tmp.name)

# Create a temporary config file at the expected location
@pytest.fixture
def temp_config_file(tmp_path):
    """Fixture to create a temporary config file."""
    temp_file = tmp_path / "test_config.yaml"
    
    with open(temp_file, "w") as f:
        json.dump(MOCK_CONFIG, f)
    
    return temp_file

@pytest.fixture
def setup_config_loader(temp_config_file):
    """Fixture to setup the ConfigLoader singleton with a temporary config file."""
    # Reset the singleton before the test
    ConfigLoader.reset()
    
    # Get the singleton instance
    config_instance = ConfigLoader.get_instance()
    # Load the config using the full temporary file path
    loaded_config = config_instance.load_config(config_path=temp_config_file)
    logger.info(f"Loaded temporary config from {temp_config_file}")
    
    yield loaded_config # Provide the loaded config to the test
    
    # Clean up after the test by resetting the singleton
    ConfigLoader.reset()
    logger.info("Reset ConfigLoader singleton.")

# Now we can safely import modules that depend on config
from database.faiss_index import FaissIndex

"""
Configuration file for pytest.
This file is automatically recognized by pytest and can define fixtures,
hooks, and other test setup that should be available across multiple test files.
"""

@pytest.fixture
def mock_config():
    """Fixture that provides a mock configuration dictionary for testing."""
    return MOCK_CONFIG.copy()

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
    # Create our mock index class
    class MockIndex(MagicMock):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.ntotal = 0
            
    # Create the mock with proper spec
    mock_index = MockIndex(spec=['add', 'search', 'ntotal'])
    
    # Set up the methods with simple return values
    mock_index.add = MagicMock()
    # Ensure the returned distances and indices match the expected shape for top_k=3
    mock_index.search = MagicMock(return_value=(np.array([[0.1, 0.2, 0.3]]), np.array([[1, 2, 3]])))
    mock_index.ntotal = 3 # Ensure ntotal reflects the mocked search results size
    
    # Mock the faiss module and its types
    mocker.patch("faiss.read_index", return_value=mock_index)
    mocker.patch("faiss.write_index")
    
    return mock_index

@pytest.fixture
def mock_db_manager(mocker):
    """Fixture to mock DatabaseManager for FAISS index tests."""
    # Create a simple mock manager without complex method tracking
    mock_manager = MagicMock(spec=['__enter__'])
    mock_cursor = MagicMock(spec=['executemany', 'execute', 'fetchall'])
    
    # Set up the methods with simple return values
    mock_manager.__enter__.return_value = mock_cursor
    mock_cursor.executemany = MagicMock()
    mock_cursor.execute = MagicMock()
    mock_cursor.fetchall = MagicMock(return_value=[])
    
    mocker.patch('database.faiss_index.DatabaseManager', return_value=mock_manager)
    
    return mock_manager

@pytest.fixture
def faiss_index(mocker, mock_faiss_index, temp_index_path, setup_config_loader):
    """Fixture for an initialized FaissIndex instance forced to use the mock index."""
    # Patch the initialization logic within FaissIndex to *always* return the mock
    mocker.patch("database.faiss_index.FaissIndex._initialize_index", return_value=mock_faiss_index)

    # Instantiate FaissIndex. It will now use the patched _initialize_index.
    # Pass temp_index_path to __init__ as it's expected.
    index = FaissIndex(index_path=temp_index_path)

    # If _initialize_index normally sets other attributes like is_hnsw,
    # we might need to set them manually here based on the mock.
    # Let's assume mock_faiss_index isn't HNSW for now.
    index.is_hnsw = False # Adjust if mock represents HNSW

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
    
    # Create simple mocks without complex tracking
    mock_fetcher = MagicMock()
    mock_fetcher.download_arxiv_pdf.return_value = "downloaded/path.pdf"
    mocker.patch('main.ArxivPaperFetcher', return_value=mock_fetcher)
    
    mock_extractor = MagicMock()
    mock_extractor.process_pdf.return_value = {"text_data": {"1": "Sample text"}}
    mocker.patch('main.PDFExtractor', return_value=mock_extractor)
    
    mock_preprocessor = MagicMock()
    mock_preprocessor.clean_text.return_value = "Cleaned sample text"
    mocker.patch('main.TextPreprocessor', return_value=mock_preprocessor)
    
    mock_chunker = MagicMock()
    mock_chunker.chunk_text.return_value = ["Chunk 1", "Chunk 2"]
    mocker.patch('main.TextChunker', return_value=mock_chunker)
    
    mock_embedder = MagicMock()
    mock_embedder.generate_embeddings.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
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
        'cursor': mock_cursor,
        'faiss': mock_faiss
    }

# Fixture to mock DatabaseManager specifically for Arxiv API tests
@pytest.fixture
def mock_db_manager_arxiv():
    """Fixture to mock DatabaseManager for Arxiv API tests."""
    with patch('data_pipeline.arxiv_api.DatabaseManager') as mock_db:
        mock_cursor = MagicMock()
        mock_db.return_value.__enter__.return_value = mock_cursor
        mock_db.return_value.__exit__.return_value = None
        yield mock_cursor # Yield the cursor for tests to use

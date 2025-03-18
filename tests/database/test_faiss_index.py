import pytest
import faiss
import numpy as np
from unittest.mock import MagicMock

from database.faiss_index import FaissIndex


def test_initialize_index_creates_new_if_not_found(mocker):
    """Test index creation when no saved index exists."""
    mocker.patch("faiss.read_index", side_effect=Exception("File not found"))
    
    # Mock FAISS index creation
    mock_index = mocker.MagicMock()
    mocker.patch("faiss.IndexHNSWFlat", return_value=mock_index)
    mocker.patch("faiss.IndexFlatL2", return_value=mock_index)

    index = FaissIndex()
    
    assert index.index is not None  # Ensures index is initialized
    assert isinstance(index.index, MagicMock)  # Should be a mock object

def test_initialize_index_loads_existing(mock_faiss_index):
    """Test loading an existing FAISS index."""
    index = FaissIndex()
    assert index.index == mock_faiss_index

def test_add_embeddings_correct_shape(faiss_index, mock_faiss_index, mocker):
    """Test adding embeddings with correct shape."""
    # Mock the DatabaseManager to prevent actual database operations
    mock_db_context = MagicMock()
    mock_cursor = MagicMock()
    mock_db_context.__enter__.return_value = mock_cursor
    mocker.patch('database.faiss_index.DatabaseManager', return_value=mock_db_context)
    
    embeddings = np.random.rand(10, faiss_index.dim).astype(np.float32)
    # Create matching chunk_ids for the embeddings
    chunk_ids = list(range(1, 11))  # 10 chunk IDs to match 10 embeddings
    
    faiss_index.add_embeddings(embeddings, chunk_ids)
    mock_faiss_index.add.assert_called_once()  # Ensure FAISS add() was called

def test_add_embeddings_dimension_mismatch(faiss_index):
    """Test adding embeddings with wrong shape raises an error."""
    embeddings = np.random.rand(10, faiss_index.dim - 1).astype(np.float32)
    chunk_ids = list(range(1, 11))  # 10 chunk IDs to match 10 embeddings
    
    with pytest.raises(AssertionError, match="Embedding dimension mismatch!"):
        faiss_index.add_embeddings(embeddings, chunk_ids)

def test_add_embeddings_count_mismatch(faiss_index):
    """Test adding embeddings with mismatched count of chunk_ids raises an error."""
    embeddings = np.random.rand(10, faiss_index.dim).astype(np.float32)
    chunk_ids = list(range(1, 9))  # Only 8 chunk IDs for 10 embeddings
    
    with pytest.raises(AssertionError, match="Number of embeddings must match number of chunk IDs"):
        faiss_index.add_embeddings(embeddings, chunk_ids)

def test_save_index_calls_faiss_write(faiss_index, mock_faiss_index):
    """Ensure saving index calls faiss.write_index()."""
    faiss_index.save_index()
    faiss.write_index.assert_called_once_with(mock_faiss_index, faiss_index.index_path)

def test_search_returns_indices_and_distances(faiss_index, mock_faiss_index):
    """Test search returns expected indices and distances."""
    query_embedding = np.random.rand(1, faiss_index.dim).astype(np.float32)
    indices, distances = faiss_index.search(query_embedding)
    assert isinstance(indices, list)  # Now returns a list of chunk_ids
    assert isinstance(distances, np.ndarray)
    assert len(indices) <= 3  # Mocked search returns up to 3 results
    assert distances.shape == (3,)

def test_search_with_dimension_mismatch(faiss_index):
    """Test querying with incorrect embedding shape raises an error."""
    query_embedding = np.random.rand(1, faiss_index.dim - 1).astype(np.float32)
    with pytest.raises(AssertionError, match="Query embedding dimension mismatch!"):
        faiss_index.search(query_embedding)

# Tests for insert_chunks_into_db method
def test_insert_chunks_into_db(faiss_index, mocker):
    """Test inserting chunks into database and FAISS index."""
    # Mock the database operations
    mock_db_context = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.lastrowid = 42  # Simulate auto-incrementing IDs
    mock_db_context.__enter__.return_value = mock_cursor
    mocker.patch('database.faiss_index.DatabaseManager', return_value=mock_db_context)
    
    # Mock the embedding model
    mock_embedding_model = mocker.MagicMock()
    mock_embedding = np.random.rand(2, faiss_index.dim).astype(np.float32)  # Shape for 2 chunks
    mock_embedding_model.return_value = mock_embedding
    
    # Mock add_embeddings to verify it's called correctly
    mocker.patch.object(faiss_index, 'add_embeddings')
    
    # Test data
    chunks = ["This is chunk 1", "This is chunk 2"]
    paper_id = "1234.56789"
    
    # Call the method
    faiss_index.insert_chunks_into_db(chunks, paper_id, mock_embedding_model)
    
    # Verify database operations
    assert mock_cursor.execute.call_count == 2  # One call per chunk
    
    # Verify the SQL query and parameters
    expected_calls = [
        mocker.call(
            "INSERT INTO paper_chunks (paper_id, chunk_text) VALUES (?, ?)",
            (paper_id, "This is chunk 1")
        ),
        mocker.call(
            "INSERT INTO paper_chunks (paper_id, chunk_text) VALUES (?, ?)",
            (paper_id, "This is chunk 2")
        )
    ]
    mock_cursor.execute.assert_has_calls(expected_calls, any_order=False)
    
    # Verify embedding model was called once with all chunks
    assert mock_embedding_model.call_count == 1
    mock_embedding_model.assert_called_once_with(chunks)
    
    # Verify add_embeddings was called with correct parameters
    faiss_index.add_embeddings.assert_called_once()
    call_args = faiss_index.add_embeddings.call_args[0]
    assert len(call_args) == 2
    embeddings_arg, chunk_ids_arg = call_args
    
    # Check embeddings shape and type
    assert embeddings_arg.shape == (2, faiss_index.dim)
    assert embeddings_arg.dtype == np.float32
    
    # Check chunk IDs
    assert chunk_ids_arg == [42, 42]  # Both chunks get the same ID in our mock

def test_insert_chunks_into_db_empty_list(faiss_index, mocker):
    """Test inserting an empty list of chunks."""
    # Mock the database operations
    mock_db_context = MagicMock()
    mock_cursor = MagicMock()
    mock_db_context.__enter__.return_value = mock_cursor
    mocker.patch('database.faiss_index.DatabaseManager', return_value=mock_db_context)
    
    # Mock the embedding model
    mock_embedding_model = mocker.MagicMock()
    
    # Mock add_embeddings to verify it's called correctly
    mocker.patch.object(faiss_index, 'add_embeddings')
    
    # Test with empty chunks list
    chunks = []
    paper_id = "1234.56789"
    
    # Call the method
    faiss_index.insert_chunks_into_db(chunks, paper_id, mock_embedding_model)
    
    # Verify no database operations occurred
    mock_cursor.execute.assert_not_called()
    
    # Verify embedding model was not called
    mock_embedding_model.assert_not_called()
    
    # Verify add_embeddings was not called
    faiss_index.add_embeddings.assert_not_called()

def test_insert_chunks_into_db_embedding_error(faiss_index, mocker):
    """Test handling of errors during embedding generation."""
    # Mock the database operations
    mock_db_context = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.lastrowid = 42
    mock_db_context.__enter__.return_value = mock_cursor
    mocker.patch('database.faiss_index.DatabaseManager', return_value=mock_db_context)
    
    # Mock the embedding model to raise an exception
    mock_embedding_model = mocker.MagicMock(side_effect=ValueError("Embedding error"))
    
    # Test data
    chunks = ["This is chunk 1"]
    paper_id = "1234.56789"
    
    # Call the method and expect it to raise the error
    with pytest.raises(ValueError, match="Embedding error"):
        faiss_index.insert_chunks_into_db(chunks, paper_id, mock_embedding_model)
    
    # Verify database operations occurred before the error
    mock_cursor.execute.assert_called_once()

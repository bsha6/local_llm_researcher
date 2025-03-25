import pytest
import numpy as np
import sqlite3
from unittest.mock import patch, MagicMock

from database.faiss_search import FaissSearcher


class TestFaissSearcher:
    
    @pytest.fixture
    def mock_sqlite_connection(self, monkeypatch):
        """Fixture to mock SQLite connection and cursor."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        
        # Mock the fetchone method to return sample data
        mock_cursor.fetchone.side_effect = [
            ("This is chunk 1",),
            ("This is chunk 2",),
            ("This is chunk 3",),
            None  # For testing when no result is found
        ]
        
        # Patch sqlite3.connect to return our mock connection
        monkeypatch.setattr(sqlite3, "connect", lambda _: mock_conn)
        
        return mock_conn, mock_cursor
    
    def test_init(self, faiss_index, mocker):
        """Test initialization of FaissSearcher."""
        # Mock the config
        mock_config = {"database": {"arxiv_db_path": ":memory:"}}
        mocker.patch('database.faiss_search.FaissSearcher.get_config', return_value=mock_config)
        
        searcher = FaissSearcher(faiss_index)
        
        assert searcher.faiss_index == faiss_index
        assert searcher.db == ":memory:"
    
    def test_search(self, faiss_index, mock_sqlite_connection, mocker):
        """Test the search method."""
        # Mock the config
        mock_config = {"database": {"arxiv_db_path": ":memory:"}}
        mocker.patch('database.faiss_search.FaissSearcher.get_config', return_value=mock_config)
        
        searcher = FaissSearcher(faiss_index)
        
        # Create a mock query embedding
        query_embedding = np.random.rand(1, 384).astype(np.float32)
        
        # Call the search method
        results = searcher.search(query_embedding, top_k=3)
        
        # Verify the search was performed with the right parameters
        faiss_index.index.search.assert_called_once()
        args, kwargs = faiss_index.index.search.call_args
        assert np.array_equal(args[0], query_embedding)
        assert args[1] == 3
        
        # Verify the results
        assert len(results) == 3
        assert results[0] == "This is chunk 1"
        assert results[1] == "This is chunk 2"
        assert results[2] == "This is chunk 3"
    
    def test_fetch_metadata(self, faiss_index, mock_sqlite_connection, mocker):
        """Test the _fetch_metadata method."""
        # Mock the config
        mock_config = {"database": {"arxiv_db_path": ":memory:"}}
        mocker.patch('database.faiss_search.FaissSearcher.get_config', return_value=mock_config)
        
        mock_conn, mock_cursor = mock_sqlite_connection
        searcher = FaissSearcher(faiss_index)
        
        # Call the _fetch_metadata method with some indices
        results = searcher._fetch_metadata([0, 1, 2, 3])
        
        # Verify the database was queried correctly
        assert mock_cursor.execute.call_count == 4
        
        # Check the SQL queries
        for i in range(4):
            args, kwargs = mock_cursor.execute.call_args_list[i]
            assert args[0] == "SELECT chunk_text FROM paper_chunks WHERE faiss_idx=?"
            assert args[1] == (i+1,)
        
        # Verify the results
        assert len(results) == 3  # Only 3 results because the 4th fetchone returns None
        assert results[0] == "This is chunk 1"
        assert results[1] == "This is chunk 2"
        assert results[2] == "This is chunk 3"
    
    def test_search_dimension_mismatch(self, faiss_index, mocker):
        """Test that search raises an assertion error when dimensions don't match."""
        # Mock the config
        mock_config = {"database": {"arxiv_db_path": ":memory:"}}
        mocker.patch('database.faiss_search.FaissSearcher.get_config', return_value=mock_config)
        
        searcher = FaissSearcher(faiss_index)
        
        # Create a query embedding with wrong dimensions
        query_embedding = np.random.rand(1, 128).astype(np.float32)  # Wrong dimension (should be 384)
        
        # Verify that an assertion error is raised
        with pytest.raises(AssertionError, match="Query embedding dimension mismatch!"):
            searcher.search(query_embedding)
    
    def test_empty_results(self, faiss_index, mocker):
        """Test behavior when no results are found."""
        # Mock the config
        mock_config = {"database": {"arxiv_db_path": ":memory:"}}
        mocker.patch('database.faiss_search.FaissSearcher.get_config', return_value=mock_config)
        
        # Mock the search method to return empty results
        faiss_index.index.search.return_value = (np.array([[]]), np.array([[]]))
        
        searcher = FaissSearcher(faiss_index)
        
        # Create a mock query embedding
        query_embedding = np.random.rand(1, 384).astype(np.float32)
        
        # Mock _fetch_metadata to return empty list
        with patch.object(searcher, '_fetch_metadata', return_value=[]):
            results = searcher.search(query_embedding)
            
            # Verify the results are empty
            assert len(results) == 0
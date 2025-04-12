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
        
        # --- Mock the underlying FAISS index search method --- 
        mock_distances = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
        mock_faiss_indices = np.array([[10, 20, 30]], dtype=np.int64) # Use distinct faiss indices
        mocker.patch.object(
            faiss_index.index,
            'search',
            return_value=(mock_distances, mock_faiss_indices)
        )
        # -----------------------------------------------------
        
        searcher = FaissSearcher(faiss_index)
        
        # Create a mock query embedding
        query_embedding = np.random.rand(1, faiss_index.dim).astype(np.float32) # Use faiss_index.dim
        
        # Call the search method
        results = searcher.search(query_embedding, top_k=3)
        
        # Verify the search was performed with the right parameters
        faiss_index.index.search.assert_called_once_with(query_embedding, 3)
        
        # Verify the DB was queried correctly by _fetch_metadata
        mock_conn, mock_cursor = mock_sqlite_connection
        # The search method calls _fetch_metadata with the faiss_indices from the mocked search return
        expected_faiss_indices = mock_faiss_indices.flatten().tolist()
        assert mock_cursor.execute.call_count == len(expected_faiss_indices)
        # Check the first call as an example (assuming fetchone returns the mocked chunks)
        sql_query, params = mock_cursor.execute.call_args_list[0][0]
        assert sql_query == "SELECT chunk_text FROM paper_chunks WHERE faiss_idx=?"
        # Convert from 0-indexed FAISS indices to 1-indexed SQLite indices
        assert params == (10+1,) # chunk_id corresponding to faiss_idx 10 (from mock_sqlite_connection)
        
        # Verify the results (using the mocked fetchone results)
        assert len(results) == 3
        assert results[0] == "This is chunk 1" # Corresponds to fetchone call for faiss_idx 10
        assert results[1] == "This is chunk 2" # Corresponds to fetchone call for faiss_idx 20
        assert results[2] == "This is chunk 3" # Corresponds to fetchone call for faiss_idx 30
    
    def test_fetch_metadata(self, faiss_index, mock_sqlite_connection, mocker):
        """Test the _fetch_metadata method."""
        # Mock the config
        mock_config = {"database": {"arxiv_db_path": ":memory:"}}
        mocker.patch('database.faiss_search.FaissSearcher.get_config', return_value=mock_config)
        
        mock_conn, mock_cursor = mock_sqlite_connection
        searcher = FaissSearcher(faiss_index)
        
        # Call the _fetch_metadata method with some faiss indices
        faiss_ids_to_fetch = [1, 2, 3, 4] 
        results = searcher._fetch_metadata(faiss_ids_to_fetch)
        
        # Verify the database was queried correctly
        assert mock_cursor.execute.call_count == len(faiss_ids_to_fetch)
        
        # Check the SQL queries
        for i, faiss_idx in enumerate(faiss_ids_to_fetch):
            args, kwargs = mock_cursor.execute.call_args_list[i]
            assert args[0] == "SELECT chunk_text FROM paper_chunks WHERE faiss_idx=?"
            assert args[1] == (faiss_idx+1,) # Convert from 0-indexed FAISS indices to 1-indexed SQLite indices
        
        # Verify the results
        assert len(results) == 3  # Only 3 results because the 4th fetchone returns None
    
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
        
        # --- Mock the underlying FAISS index search method to return empty --- 
        mocker.patch.object(
            faiss_index.index,
            'search',
            return_value=(np.array([[]], dtype=np.float32), np.array([[]], dtype=np.int64))
        )
        # -------------------------------------------------------------------
        
        searcher = FaissSearcher(faiss_index)
        
        # Create a mock query embedding
        query_embedding = np.random.rand(1, faiss_index.dim).astype(np.float32)
        
        # Call search
        results = searcher.search(query_embedding)

        # Verify faiss search was called
        faiss_index.index.search.assert_called_once()

        # Verify _fetch_metadata was NOT called (or called with empty list)
        # Since the mocked search returns empty indices, _fetch_metadata should receive []
        # Let's mock _fetch_metadata to assert it gets called with an empty list
        with patch.object(searcher, '_fetch_metadata') as mock_fetch:
            results = searcher.search(query_embedding) # Call again with mock active
            mock_fetch.assert_called_once_with([])

        # Verify the final results are empty
        assert len(results) == 0
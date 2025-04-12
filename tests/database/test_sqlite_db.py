import pytest
from unittest.mock import patch, MagicMock
import sqlite3

from database.sqlite_db import DatabaseManager

class TestDatabaseManager:
    
    @pytest.fixture(autouse=True)
    def setup_db(self, setup_config_loader):
        """Setup and cleanup for each test, ensuring config is loaded."""
        # Config is now loaded by setup_config_loader fixture before this runs
        self.db_path = ":memory:"
        self.db = DatabaseManager(self.db_path)
        yield
        # Cleanup
        if hasattr(self, 'db'):
            try:
                self.db._conn.close()
            except (sqlite3.Error, AttributeError):
                pass
    
    @pytest.fixture
    def mock_sqlite3_connect(self):
        """Fixture to mock sqlite3.connect."""
        with patch('sqlite3.connect') as mock_connect:
            # Create mock connection and cursor
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            
            # Configure the mock connection to return the mock cursor
            mock_conn.cursor.return_value = mock_cursor
            
            # Configure connect to return the mock connection
            mock_connect.return_value = mock_conn
            
            yield {
                'connect': mock_connect,
                'conn': mock_conn,
                'cursor': mock_cursor
            }
    
    def test_init_db(self, mock_sqlite3_connect):
        """Test database initialization."""
        mock_cursor = mock_sqlite3_connect['cursor']
        
        # Initialize database
        self.db.init_db()
        
        # Verify that all required tables were created
        expected_tables = [
            'papers',
            'query_history',
            'paper_chunks'
        ]
        
        # Check that CREATE TABLE was called for each table
        assert mock_cursor.execute.call_count >= len(expected_tables)
        
        # Verify commit was called
        mock_sqlite3_connect['conn'].commit.assert_called()
    
    def test_fetch_metadata_with_params(self, mock_sqlite3_connect):
        """Test _fetch_metadata with query parameters."""
        # Setup
        test_query = "SELECT * FROM test_table WHERE id = ?"
        test_params = (1,)
        expected_result = [(1, "test_data")]
        
        # Configure mock to return expected result
        mock_cursor = mock_sqlite3_connect['cursor']
        mock_cursor.fetchall.return_value = expected_result
        
        # Execute
        result = DatabaseManager._fetch_metadata(self.db_path, test_query, test_params)
        
        # Assert
        assert result == expected_result
        mock_cursor.execute.assert_called_once_with(test_query, test_params)
        mock_sqlite3_connect['conn'].commit.assert_called_once()
        mock_sqlite3_connect['conn'].close.assert_called_once()
    
    def test_fetch_metadata_without_params(self, mock_sqlite3_connect):
        """Test _fetch_metadata without query parameters."""
        # Setup
        test_query = "SELECT * FROM test_table"
        expected_result = [(1, "test_data"), (2, "more_data")]
        
        # Configure mock to return expected result
        mock_cursor = mock_sqlite3_connect['cursor']
        mock_cursor.fetchall.return_value = expected_result
        
        # Execute
        result = DatabaseManager._fetch_metadata(self.db_path, test_query)
        
        # Assert
        assert result == expected_result
        mock_cursor.execute.assert_called_once_with(test_query, ())
        mock_sqlite3_connect['conn'].commit.assert_called_once()
        mock_sqlite3_connect['conn'].close.assert_called_once()
    
    def test_fetch_metadata_empty_result(self, mock_sqlite3_connect):
        """Test _fetch_metadata when no results are returned."""
        # Setup
        test_query = "SELECT * FROM empty_table"
        expected_result = []
        
        # Configure mock to return empty result
        mock_cursor = mock_sqlite3_connect['cursor']
        mock_cursor.fetchall.return_value = expected_result
        
        # Execute
        result = DatabaseManager._fetch_metadata(self.db_path, test_query)
        
        # Assert
        assert result == expected_result
        assert len(result) == 0
        mock_cursor.execute.assert_called_once_with(test_query, ())
    
    def test_context_manager(self, mock_sqlite3_connect):
        """Test the context manager functionality."""        
        with DatabaseManager(self.db_path) as cursor:
            cursor.execute("SELECT 1")
            
        # Verify connection was properly managed
        mock_sqlite3_connect['conn'].commit.assert_called_once()
        mock_sqlite3_connect['conn'].close.assert_called_once()
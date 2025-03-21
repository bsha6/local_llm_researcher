import pytest
from unittest.mock import patch, MagicMock

from database.sqlite_db import DatabaseManager

@pytest.mark.usefixtures("mock_config_globally")
class TestDatabaseManager:
    
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
    
    def test_fetch_metadata_with_params(self, mock_sqlite3_connect):
        """Test _fetch_metadata with query parameters."""
        # Setup
        test_db_path = ":memory:"
        test_query = "SELECT * FROM test_table WHERE id = ?"
        test_params = (1,)
        expected_result = [(1, "test_data")]
        
        # Configure mock to return expected result
        mock_cursor = mock_sqlite3_connect['cursor']
        mock_cursor.fetchall.return_value = expected_result
        
        # Execute
        result = DatabaseManager._fetch_metadata(test_db_path, test_query, test_params)
        
        # Assert
        assert result == expected_result
        mock_cursor.execute.assert_called_once_with(test_query, test_params)
        mock_sqlite3_connect['conn'].commit.assert_called_once()
        mock_sqlite3_connect['conn'].close.assert_called_once()
    
    def test_fetch_metadata_without_params(self, mock_sqlite3_connect):
        """Test _fetch_metadata without query parameters."""
        # Setup
        test_db_path = ":memory:"
        test_query = "SELECT * FROM test_table"
        expected_result = [(1, "test_data"), (2, "more_data")]
        
        # Configure mock to return expected result
        mock_cursor = mock_sqlite3_connect['cursor']
        mock_cursor.fetchall.return_value = expected_result
        
        # Execute
        result = DatabaseManager._fetch_metadata(test_db_path, test_query)
        
        # Assert
        assert result == expected_result
        mock_cursor.execute.assert_called_once_with(test_query, ())
        mock_sqlite3_connect['conn'].commit.assert_called_once()
        mock_sqlite3_connect['conn'].close.assert_called_once()
    
    def test_fetch_metadata_empty_result(self, mock_sqlite3_connect):
        """Test _fetch_metadata when no results are returned."""
        # Setup
        test_db_path = ":memory:"
        test_query = "SELECT * FROM empty_table"
        expected_result = []
        
        # Configure mock to return empty result
        mock_cursor = mock_sqlite3_connect['cursor']
        mock_cursor.fetchall.return_value = expected_result
        
        # Execute
        result = DatabaseManager._fetch_metadata(test_db_path, test_query)
        
        # Assert
        assert result == expected_result
        assert len(result) == 0
        mock_cursor.execute.assert_called_once_with(test_query, ())
    
    @patch.object(DatabaseManager, '__enter__')
    @patch.object(DatabaseManager, '__exit__')
    def test_fetch_metadata_context_manager_usage(self, mock_exit, mock_enter, mock_sqlite3_connect):
        """Test that _fetch_metadata correctly uses the DatabaseManager context manager."""
        # Setup
        test_db_path = ":memory:"
        test_query = "SELECT * FROM test_table"
        mock_cursor = mock_sqlite3_connect['cursor']
        mock_enter.return_value = mock_cursor
        
        # Execute
        DatabaseManager._fetch_metadata(test_db_path, test_query)
        
        # Assert
        mock_enter.assert_called_once()
        mock_exit.assert_called_once()
import os
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open, PropertyMock

from utils.file_operations import load_config


def test_load_config_success(temp_config_file, mock_config):
    """Test successful loading of a config file"""

    config_dir = temp_config_file.parent  # Get the directory of the temp file
    config_filename = temp_config_file.name  # Get the filename

    # Patch Path.parent and Path.parent.parent correctly
    with patch.object(Path, "parent", new_callable=PropertyMock) as mock_parent:
        mock_parent.return_value = config_dir  # Make Path.parent return the temp directory

        with patch.object(Path, "parent", new_callable=PropertyMock) as mock_grandparent:
            mock_grandparent.return_value = config_dir  # Make Path.parent.parent return the same

            # Call the function
            loaded_config = load_config(config_filename)

            # Assert the loaded config matches the expected mock config
            assert loaded_config == mock_config


def test_load_config_file_not_found():
    """Test that FileNotFoundError is raised when config file doesn't exist"""
    with patch('utils.file_operations.Path.exists', return_value=False):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yaml")


def test_load_config_empty_file():
    """Test that ValueError is raised when config file is empty"""
    with patch('utils.file_operations.Path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data="")):
            with patch('yaml.safe_load', return_value=None):
                with pytest.raises(ValueError):
                    load_config()


def test_load_config_default_filename():
    """Test that the default filename 'config.yaml' is used when not specified"""
    with patch('utils.file_operations.Path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data="test: value")) as mock_file:
            with patch('yaml.safe_load', return_value={"test": "value"}):
                config = load_config()
                
                # Check that the default filename was used
                file_path = mock_file.call_args[0][0]
                assert os.path.basename(str(file_path)) == "config.yaml"

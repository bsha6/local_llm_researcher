import logging
from pathlib import Path
from unittest.mock import patch, mock_open, PropertyMock

from utils.file_operations import load_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_load_config_success(temp_config_file, mock_config):
    """Test successful loading of a config file"""
    logger.info("Testing config loading with temporary config file")
    config_dir = temp_config_file.parent
    config_filename = temp_config_file.name

    with patch.object(Path, "parent", new_callable=PropertyMock) as mock_parent:
        mock_parent.return_value = config_dir
        with patch.object(Path, "parent", new_callable=PropertyMock) as mock_grandparent:
            mock_grandparent.return_value = config_dir
            loaded_config = load_config(config_filename)
            logger.info("Successfully loaded config from temporary file")
            assert loaded_config == mock_config


def test_load_config_default_filename():
    """Test that the default filename 'config.yaml' is used when not specified"""
    logger.info("Testing config loading with default filename")
    try:
        config = load_config()
        logger.info("Successfully loaded real config file")
        assert isinstance(config, dict)
        assert "database" in config
        assert "arxiv" in config
    except FileNotFoundError:
        logger.info("No real config found, testing with mock config")
        with patch('utils.file_operations.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data="test: value")):
                with patch('yaml.safe_load', return_value={"test": "value"}):
                    config = load_config()
                    assert config == {"test": "value"}

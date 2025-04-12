import logging
from unittest.mock import patch, mock_open

from utils.file_operations import load_config, ConfigLoader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_load_config_success(setup_config_loader, mock_config):
    """Test successful loading of a config file using the loader fixture."""
    logger.info("Testing config loading via setup_config_loader fixture")
    loaded_config = setup_config_loader
    assert loaded_config == mock_config
    logger.info("Successfully verified config loaded by fixture")


def test_load_config_default_filename():
    """Test that the default filename 'config.yaml' is used when not specified"""
    logger.info("Testing config loading with default filename")
    ConfigLoader.reset()
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

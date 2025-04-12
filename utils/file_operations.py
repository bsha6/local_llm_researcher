import yaml
from pathlib import Path
from typing import Optional, Dict

class ConfigLoader:
    _instance: Optional['ConfigLoader'] = None
    _config: Optional[Dict] = None

    @classmethod
    def get_instance(cls) -> 'ConfigLoader':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset the singleton instance (useful for testing)"""
        cls._instance = None
        cls._config = None

    def load_config(self, config_path: Optional[Path] = None) -> Dict:
        """Loads the configuration file.
        
        Args:
            config_path (Optional[Path]): Path to the config file. If None, defaults to 'config.yaml' in the project's root directory.
        
        Returns:
            dict: Parsed configuration data.
        
        Raises:
            FileNotFoundError: If the config file does not exist.
            ValueError: If the config file is empty or malformed.
        """
        if self._config is not None:
            return self._config

        if config_path is None:
            script_dir = Path(__file__).parent.parent
            config_path = script_dir / "config.yaml"
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as config_file:
            config = yaml.safe_load(config_file)
            if not config:
                raise ValueError(f"Config file is empty or invalid: {config_path}")

        self._config = config
        return config

def load_config(config_path: Optional[Path] = None) -> Dict:
    """Helper function to get config using the singleton loader.
    This maintains backwards compatibility with existing code."""
    return ConfigLoader.get_instance().load_config(config_path)

import yaml
from pathlib import Path

def load_config(config_filename="config.yaml"):
    """Loads the configuration file from the project's root directory.
    
    Args:
        config_filename (str): Name of the config file. Defaults to 'config.yaml'.
    
    Returns:
        dict: Parsed configuration data.
    
    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config file is empty or malformed.
    """
    # Get the absolute path to the project's root directory
    script_dir = Path(__file__).parent.parent
    config_path = script_dir / config_filename

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load configuration
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
        if not config:
            raise ValueError(f"Config file is empty or invalid: {config_path}")

    return config

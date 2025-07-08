import os
import yaml
import logging as logger

def load_config(config_file: str) -> dict:       

    """
    This class allows us to extract the configuration of the 6x2pt likelihood
    form a YAML file.
    
    config_file: str, the configuration file of format .yaml; located in the same directory as this script.

    Return:
        config_data: dict, the configuration data loaded from the YAML file.
    """

    # Configure the logger:
    logger.basicConfig(
        level=logger.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # The YAML file must be found in the current directory, defined as:
    current_dir = os.path.dirname(os.path.abspath(__file__)) + '/' # where __file__ is the current file and its path/directory

    print(f"Current directory: {current_dir}")
    print(f"Config file: {config_file}")

    # Try to open the YAML file provided as config_file
    with open(os.path.join(current_dir, config_file), 'r') as yamlfile:
        try:
            config_data = yaml.load(yamlfile, Loader=yaml.FullLoader)
        except Exception as e:
            logger.error(f"Error loading YAML file with name{config_file} in directory {current_dir}: \n {e}")
            return None
        
    return config_data


# dir = load_config('6x2pt_config.yaml')

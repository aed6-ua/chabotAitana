import logging, json

# Load the configuration file
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

# Accessing the configuration
config = load_config("config.json")

log = config["global"]["log"]
verbose = config["global"]["verbose"]

# Configure logging
logging.basicConfig(level=logging.INFO, filename='server.log', filemode='a', format='%(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

def log_info(text):
    if log=="Y":
        logger.info(text)
    if verbose=="Y":
        print(text)

def log_error(text):
    if log=="Y":
        logger.error(text)
    if verbose=="Y":
        print(text)

def show_info(text):
    if verbose=="Y":
        print(text)
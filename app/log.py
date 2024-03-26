import logging, json

# Configure logging
logging.basicConfig(level=logging.INFO, filename='server.log', filemode='a', format='%(name)s - %(levelname)s - %(message)s')

class Log:
    def __init__(self, config_path):
        # Accessing the configuration
        self.load_config(config_path)
        self.log = self.config["global"]["log"]
        self.verbose = self.config["global"]["verbose"]

        self.logger = logging.getLogger(__name__)

    def getConfig(self):
        return self.config
    
    def info(self, text):
        if self.log=="Y":
            self.logger.info(text)
        if self.verbose=="Y":
            print(text)

    def error(self, text):
        if self.log=="Y":
            self.logger.error(text)
        if self.verbose=="Y":
            print(text)

    def verbose(self, text):
        if self.verbose=="Y":
            print(text)

    # Load the configuration file
    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            self.config = json.load(file)

# Creating the log object
if 'logger' not in globals():
    logger = Log("config.json")

if 'config' not in globals():
    config = logger.getConfig()


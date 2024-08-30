import logging, json

class Log:
    def __init__(self, config_path):
        # Accessing the configuration
        self.load_config(config_path)
        self.log = self.config["global"]["log"]
        self.verbose = self.config["global"]["verbose"]
        self.ID = self.config["global"]["id"]

        # Configure logging
        log_filename = self.config["global"]["log_folder"] + 'server.log'
        logging.basicConfig(level=logging.INFO, filename=log_filename, datefmt='%Y/%m/%d %H:%M:%S', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')

        self.logger = logging.getLogger(__name__)

    def getConfig(self):
        return self.config
    
    def info(self, text):
        text = "(" + self.ID + ") " + text
        if self.log=="Y":
            self.logger.info(text)
        if self.verbose=="Y":
            print(text)

    def error(self, text):
        text = "(" + self.ID + ") " + text
        if self.log=="Y":
            self.logger.error(text)
        if self.verbose=="Y":
            print(text)

    def verbose(self, text):
        if self.verbose=="Y":
            text = "(" + self.ID + ") " + text
            print(text)

    # Load the configuration file
    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            self.config = json.load(file)

if 'logger' not in globals():
    logger=None

if 'config' not in globals():
    config=None

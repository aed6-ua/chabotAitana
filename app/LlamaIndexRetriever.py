from abc import ABC, abstractmethod
import logging
from llama_index.core import StorageContext, load_index_from_storage

from Retriever import RetrievalStrategy

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class LlamaIndexRetriever(RetrievalStrategy):
    def __init__(self, index_path="./storage"):
        storage_context = StorageContext.from_defaults(persist_dir=index_path)
        # load index
        self.index = load_index_from_storage(storage_context)
        logging.info("LlamaIndex loaded successfully.")

    def retrieve(self, query):
        # Use llamaindex to find the most relevant documents for the query
        results = self.index.search(query)
        return results

from abc import ABC, abstractmethod
import logging

from llama_index.core import StorageContext, load_index_from_storage

from Retriever import RetrievalStrategy

from env import env

if (env["log"]=="Y"):
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class LlamaIndexRetriever(RetrievalStrategy):
    def __init__(self, index_path, model_name):
        self.index_path = index_path #env["embeddings_folder"]
        self.model_name = model_name #env["embedding_model_name"]



    def load_embeddings(self):
        self.storage_context = StorageContext.from_defaults(persist_dir=self.index_path)
        self.index = load_index_from_storage(self.storage_context)
        if (env["log"]=="Y"):
            logging.info("LlamaIndex embeddings loaded successfully.")


    
    def retrieve(self, query):
        if (self.index is None):
            self.load_embeddings()

        # Use llamaindex to find the most relevant documents for the query
        results = self.index.search(query)
        return results

from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
import pickle
import logging

from Retriever import RetrievalStrategy

from env import env

if (env["log"]=="Y"):
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Implement Concrete Retrieval Strategies
class SentenceTransformerEmbeddings:
    def __init__(self, model_name='all-MiniLM-L6-v2', filename='embeddings', top_k=5):
            if (env["log"]=="Y"):
                logging.info("Initializing SentenceTransformerRetrieval...")
            
            self.model = SentenceTransformer(model_name)
            if (env["log"]=="Y"):
                logging.info(f"Loaded SentenceTransformer model: {model_name}")

            self.corpus_embeddings = []
            self.corpus_texts

            self.filename = filename
            self.top_k = top_k



    def create_embeddings(self): #TODO: fer!
        pass


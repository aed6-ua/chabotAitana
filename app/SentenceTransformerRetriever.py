from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer, util
import pickle
import logging

from Retriever import RetrievalStrategy

from env import env

if (env["log"]=="Y"):
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Implement Concrete Retrieval Strategies
class SentenceTransformerRetriever(RetrievalStrategy):
    def __init__(self, model_name, filename, top_k):
            if (env["log"]=="Y"):
                logging.info("Initializing SentenceTransformerRetrieval...")
            
            self.model = SentenceTransformer(model_name)
            if (env["log"]=="Y"):
                logging.info(f"Loaded SentenceTransformer model: {model_name}")

            self.corpus_embeddings = []
            self.corpus_texts

            self.filename = filename
            self.top_k = top_k



    def load_embeddings(self):
        if (env["log"]=="Y"):
            logging.info("Loading embeddings from file...")
        
        # Load the embeddings from the file
        with open(self.filename, 'rb') as file:
            loaded_data = pickle.load(file)
        
        # Restaura los embeddings y textos desde el diccionario
        self.corpus_embeddings = loaded_data['corpus_embeddings']
        if (env["log"]=="Y"):
            logging.info(f"Loaded {len(self.corpus_embeddings)} embeddings.")
        
        self.corpus_texts_es = loaded_data.get('corpus_texts_es', [])
        if (env["log"]=="Y"):
            logging.info(f"Loaded {len(self.corpus_texts_es)} Spanish texts.")
        
        self.corpus_texts_en = loaded_data.get('corpus_texts_en', [])
        if (env["log"]=="Y"):
            logging.info(f"Loaded {len(self.corpus_texts_en)} English texts.")



    def retrieve(self, query): #TODO: comparar amb el meu codi, eliminar chunks baix l'umbral
        # Tokenize the query
        query_embedding = self.model.encode(query, convert_to_tensor=True)

        # Compute the cosine similarity between the query and the corpus
        hits = util.semantic_search(query_embedding, self.corpus_embeddings, top_k=self.top_k)[0]
        # Format the results as a list of tuples
        results = [(self.corpus_texts[hit['corpus_id']], hit['score']) for hit in hits] #if self.corpus_texts_es else [(self.corpus_texts_en[hit['corpus_id']], hit['score']) for hit in hits]
        return results

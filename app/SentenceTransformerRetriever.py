from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer, util
import pickle
import logging

from Retriever import RetrievalStrategy

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Implement Concrete Retrieval Strategies
class SentenceTransformerRetriever(RetrievalStrategy):
    def __init__(self, model_name='all-MiniLM-L6-v2', filename='embeddings', top_k=5):
            logging.info("Initializing SentenceTransformerRetrieval...")
            self.model = SentenceTransformer(model_name)
            logging.info(f"Loaded SentenceTransformer model: {model_name}")
            self.corpus_embeddings = []
            self.corpus_texts_es = []
            self.corpus_texts_en = []
            self.filename = filename
            self.top_k = top_k
            self.load_embeddings()

    def load_embeddings(self):
        logging.info("Loading embeddings from file...")
        # Load the embeddings from the file
        with open(self.filename, 'rb') as file:
            loaded_data = pickle.load(file)
        # Restaura los embeddings y textos desde el diccionario
        self.corpus_embeddings = loaded_data['corpus_embeddings']
        logging.info(f"Loaded {len(self.corpus_embeddings)} embeddings.")
        self.corpus_texts_es = loaded_data.get('corpus_texts_es', [])
        logging.info(f"Loaded {len(self.corpus_texts_es)} Spanish texts.")
        self.corpus_texts_en = loaded_data.get('corpus_texts_en', [])
        logging.info(f"Loaded {len(self.corpus_texts_en)} English texts.")

    def retrieve(self, query):
        # Tokenize the query
        query_embedding = self.model.encode(query, convert_to_tensor=True)

        # Compute the cosine similarity between the query and the corpus
        hits = util.semantic_search(query_embedding, self.corpus_embeddings, top_k=self.top_k)[0]
        # Format the results as a list of tuples
        results = [(self.corpus_texts_es[hit['corpus_id']], hit['score']) for hit in hits] if self.corpus_texts_es else [(self.corpus_texts_en[hit['corpus_id']], hit['score']) for hit in hits]
        return results

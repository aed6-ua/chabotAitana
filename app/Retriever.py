from abc import ABC, abstractmethod
from llama_index.core import StorageContext, load_index_from_storage
from sentence_transformers import SentenceTransformer, util
import pickle

from log import config, logger

# Define the Retrieval Strategy Interface
class RetrievalStrategy(ABC):
    @abstractmethod
    def retrieve(self, query):
        pass

#############################################################################################
#############################################################################################
#############################################################################################
class Retriever:
    def __init__(self, strategy: RetrievalStrategy):
        self.strategy = strategy

    def load_embeddings(self):
        # Placeholder for loading the corpus
        pass

    def retrieve(self, query):
        """
        Retrieves information based on the given query.
      
        :param query: The query for which to retrieve information.
        :return: Retrieved information or results.
        """
        # Placeholder for retrieval logic, e.g., database lookup, web search, etc.
        return f"Information related to {query}"
    
#############################################################################################
#############################################################################################
#############################################################################################
class LlamaIndexRetriever(RetrievalStrategy):
    def __init__(self, index_path, model_name):
        self.index_path = index_path #env["embeddings_folder"]
        self.model_name = model_name #env["embedding_model_name"]

    def load_embeddings(self):
        self.storage_context = StorageContext.from_defaults(persist_dir=self.index_path)
        self.index = load_index_from_storage(self.storage_context)
        logger.info("LlamaIndex embeddings loaded successfully.")
    
    def retrieve(self, query):
        if (self.index is None):
            self.load_embeddings()

        # Use llamaindex to find the most relevant documents for the query
        results = self.index.search(query)
        return results
    
#############################################################################################
#############################################################################################
#############################################################################################
class SentenceTransformerRetriever(RetrievalStrategy):
    def __init__(self, model_name, filename, top_k):
            logger.info("Initializing SentenceTransformerRetrieval...")
            
            self.model = SentenceTransformer(model_name)
            logger.info(f"Loaded SentenceTransformer model: {model_name}")

            self.corpus_embeddings = []
            self.corpus_texts

            self.filename = filename
            self.top_k = top_k

    def load_embeddings(self):
        logger.info("Loading embeddings from file...")
        
        # Load the embeddings from the file
        with open(self.filename, 'rb') as file:
            loaded_data = pickle.load(file)
        
        # Restaura los embeddings y textos desde el diccionario
        self.corpus_embeddings = loaded_data['corpus_embeddings']
        logger.info(f"Loaded {len(self.corpus_embeddings)} embeddings.")
        
        self.corpus_texts_es = loaded_data.get('corpus_texts_es', [])
        logger.info(f"Loaded {len(self.corpus_texts_es)} Spanish texts.")
        
        self.corpus_texts_en = loaded_data.get('corpus_texts_en', [])
        logger.info(f"Loaded {len(self.corpus_texts_en)} English texts.")



    def retrieve(self, query): #TODO: comparar amb el meu codi, eliminar chunks baix l'umbral
        # Tokenize the query
        query_embedding = self.model.encode(query, convert_to_tensor=True)

        # Compute the cosine similarity between the query and the corpus
        hits = util.semantic_search(query_embedding, self.corpus_embeddings, top_k=self.top_k)[0]
        # Format the results as a list of tuples
        results = [(self.corpus_texts[hit['corpus_id']], hit['score']) for hit in hits] #if self.corpus_texts_es else [(self.corpus_texts_en[hit['corpus_id']], hit['score']) for hit in hits]
        return results


from abc import ABC, abstractmethod
from sentence_transformers import util
import pickle
import logging
from llama_index.core import StorageContext, load_index_from_storage
from Model import EmbeddingsModel

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the Retrieval Strategy Interface
class RetrievalStrategy(ABC):
    @abstractmethod
    def retrieve(self, query):
        pass

# Implement Concrete Retrieval Strategies
class SentenceTransformerRetriever(RetrievalStrategy):
    def __init__(self, model: EmbeddingsModel, filename='embeddings', top_k=5):
            logging.info("Initializing SentenceTransformerRetrieval...")
            self.model = model
            logging.info(f"Loaded SentenceTransformer model: {model}")
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
        query_embedding = self.model.run(query)

        # Compute the cosine similarity between the query and the corpus
        hits = util.semantic_search(query_embedding, self.corpus_embeddings, top_k=self.top_k)[0]
        # Format the results as a list of tuples
        results = [(self.corpus_texts_es[hit['corpus_id']], hit['score']) for hit in hits] if self.corpus_texts_es else [(self.corpus_texts_en[hit['corpus_id']], hit['score']) for hit in hits]
        return results

from llama_index.core import Settings
class LlamaIndexRetriever(RetrievalStrategy):
    def __init__(self, index_path="./storage"):
        Settings.embed_model = CustomEmbeddings()
        storage_context = StorageContext.from_defaults(persist_dir=index_path)
        # load index
        self.index = load_index_from_storage(storage_context)
        self.retriever = self.index.as_retriever()
        logging.info("LlamaIndex loaded successfully.")

    def retrieve(self, query):
        # Use llamaindex to find the most relevant documents for the query
        results = self.retriever.retrieve(query)
        return results


from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr
from sentence_transformers import SentenceTransformer
import asyncio
from typing import List


class CustomEmbeddings(BaseEmbedding):
    _model: SentenceTransformer = PrivateAttr()
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = 'hackathon-pln-es/paraphrase-spanish-distilroberta'
        self._model = SentenceTransformer('hackathon-pln-es/paraphrase-spanish-distilroberta')

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._model.encode(query, convert_to_numpy=True).tolist()

    async def _aget_query_embedding(self, query: str):
        # If the model doesn't natively support asyncio, you can use executor to run the synchronous method in an asynchronous manner
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._get_query_embedding, query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._model.encode(text, convert_to_numpy=True).tolist()

    async def _aget_text_embedding(self, text: str):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._get_text_embedding, text)

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
    


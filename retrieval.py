import logging
import pickle
from abc import ABC, abstractmethod
from typing import List
import asyncio
from sentence_transformers import util, SentenceTransformer
import requests
#from llama_index.core import StorageContext, load_index_from_storage, Settings
#from llama_index.core.embeddings import BaseEmbedding
#from llama_index.core.bridge.pydantic import PrivateAttr

VALID_RETRIEVER_TYPES = ["sentence_transformer", "llama_index"]

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Retriever(ABC):

    @abstractmethod
    def retrieve(self, query):
        """
        Retrieves information based on the given query.

        :param query: The query for which to retrieve information.
        :return: List of results.
        """
        # Placeholder for retrieval logic, e.g., database lookup, web search, etc.
        return f"Information related to {query}"
    
    @abstractmethod
    def get_config(self):
        """
        Get a JSON representation of the retriever configuration for storage. The retriever can be reconstructed using the configuration.
        """
        pass

    @staticmethod
    def factory(retriever_type, **kwargs):
        if retriever_type == "simple_transformer":
            return SimpleRetriever(**kwargs)
        elif retriever_type == "llama_index":
            return None#LlamaIndexRetriever(**kwargs)
        else:
            raise ValueError(f"Invalid retriever type: {retriever_type}. Valid types are: {VALID_RETRIEVER_TYPES}")


# Implement Concrete Retrieval Strategies
class SimpleRetriever(Retriever):
    """Simple in-memory retriever using a SentenceTransformer model and a precomputed set of embeddings."""
    def __init__(self, model, filename='embeddings', top_k=5):
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
    
    def get_config(self, config):
        return {
                "type": "SimpleRetriever",
                "local": True,
                "llm_server": self.model.get_config()["server_url"],
                "index_path": config["retriever"]["index_path"],
                "model_name": config["retriever"]["model_name"],
                "filename": self.filename,
                "top_k": self.top_k,
                "data_folder": "data",
                "number_of_documents": 5,
                "llamaindex_path": "./storage",
                "collection_name": "base"
            }
    


# Retriever using ChromaDB and Unstructured
import chromadb

class ChromaDBRetriever(Retriever):
    def __init__(self, model, collection_name, chromadb_host='localhost', chromadb_port=8000, rerank=False):
        client = chromadb.HttpClient(host=chromadb_host, port=chromadb_port)
        self.server = model
        self.collection = client.get_collection(collection_name)
        logging.info(f"ChromaDB retriever initialized for collection: {collection_name}")
        self.rerank = rerank

    def retrieve(self, query, top_k=1, where=None, where_document=None):
        print("Sending: ", query)
        response = requests.post(f"{self.server}/embed", params={"input": query})
        query_embedding = response.json()["embeddings"]
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            #where={"metadata_field": "is_equal_to_this"},
            #where_document={"$contains":"search_string"}
        )
        # We get a dictionary of lists, so we need to return a list of tuples with the document content and the score

        # Rerank the results if enabled
        if (self.rerank):
            ranking = requests.post(f"{self.server}/rank", json={"messages": [query] + results["documents"][0]})
            results = ranking.json()
            results = [result["candidate"] for result in results]
        else:
            results = results["documents"][0]
        return results

    def get_config(self, config):
        return {
                "type": "ChromaDBRetriever",
                "local": True,
                "llm_server": config["retriever"]["llm_server"],
                "index_path": config["retriever"]["index_path"],
                "model_name": config["retriever"]["model_name"],
                "filename": config["retriever"]["filename"],
                "top_k": config["retriever"]["top_k"],
                "data_folder": "data",
                "number_of_documents": 5,
                "llamaindex_path": "./storage",
                "collection_name": self.collection.name,
                "rerank": self.rerank
            }



# # Retriever using LlamaIndex
# from llama_index.core.retrievers import BaseRetriever
# from llama_index.core.indices.query.embedding_utils import get_top_k_embeddings
# from llama_index.core import QueryBundle
# from llama_index.core.schema import NodeWithScore
# from typing import List, Any, Optional

# class HybridRetriever(BaseRetriever):
#     """Hybrid retriever."""

#     def __init__(
#         self,
#         vector_index,
#         docstore,
#         similarity_top_k: int = 2,
#         out_top_k: Optional[int] = None,
#         alpha: float = 0.5,
#         **kwargs: Any,
#     ) -> None:
#         """Init params."""
#         super().__init__(**kwargs)
#         self._vector_index = vector_index
#         self._embed_model = vector_index._embed_model
#         self._retriever = vector_index.as_retriever(
#             similarity_top_k=similarity_top_k
#         )
#         self._out_top_k = out_top_k or similarity_top_k
#         self._docstore = docstore
#         self._alpha = alpha

#     def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
#         """Retrieve nodes given query."""

#         # first retrieve chunks
#         nodes = self._retriever.retrieve(query_bundle.query_str)

#         # get documents, and embedding similiaryt between query and documents

#         ## get doc embeddings
#         docs = [self._docstore.get_document(n.node.index_id) for n in nodes]
#         doc_embeddings = [d.embedding for d in docs]
#         query_embedding = self._embed_model.get_query_embedding(
#             query_bundle.query_str
#         )

#         ## compute doc similarities
#         doc_similarities, doc_idxs = get_top_k_embeddings(
#             query_embedding, doc_embeddings
#         )

#         ## compute final similarity with doc similarities and original node similarity
#         result_tups = []
#         for doc_idx, doc_similarity in zip(doc_idxs, doc_similarities):
#             node = nodes[doc_idx]
#             # weight alpha * node similarity + (1-alpha) * doc similarity
#             full_similarity = (self._alpha * node.score) + (
#                 (1 - self._alpha) * doc_similarity
#             )
#             print(
#                 f"Doc {doc_idx} (node score, doc similarity, full similarity): {(node.score, doc_similarity, full_similarity)}"
#             )
#             result_tups.append((full_similarity, node))

#         result_tups = sorted(result_tups, key=lambda x: x[0], reverse=True)
#         # update scores
#         for full_score, node in result_tups:
#             node.score = full_score

#         return [n for _, n in result_tups][:self.out_top_k]
    
# from llama_index.core.schema import MetadataMode

# class LlamaIndexRetriever(Retriever):
#     def __init__(self, index):
#         self.retriever = index.as_retriever()
#         logging.info("LlamaIndex loaded successfully.")

#     def retrieve(self, query):
#         # Use llamaindex to find the most relevant documents for the query
#         results = self.retriever.retrieve(query)
#         # We get a list of NodeWithScore objects, so we need to return a list of tuples with the node content and the score
#         results = [(result.node.get_content(MetadataMode.NONE), result.score) for result in results]
#         return results




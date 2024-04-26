from abc import ABC, abstractmethod
from llama_index.core import StorageContext, load_index_from_storage
from sentence_transformers import SentenceTransformer, util
from langchain_community.embeddings import HuggingFaceEmbeddings
from llama_index.core.postprocessor import SimilarityPostprocessor
import pickle

from log import config, logger

#############################################################################################
#############################################################################################
#############################################################################################
class Retriever:
    def __init__(self):
        pass

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
class LlamaIndexRetriever():
    def __init__(self, config, datafolder):
        self.index_path = config["vector_folder"] + datafolder
        self.model_name = config["model_name"]
        self.model_folder = config["model_folder"]
        self.top_k = config["num_chunks"]
        self.min_relevance = config["%_sim_relevance"]

    def load_embeddings(self):
        self._model = HuggingFaceEmbeddings(model_name=self.model_name, cache_folder=self.model_folder)
        self.storage_context = StorageContext.from_defaults(persist_dir=self.index_path)
        #Load index
        self.index = load_index_from_storage(self.storage_context, embed_model=self._model)
        logger.info("LlamaIndex embeddings loaded successfully.")
    
    def retrieve(self, query):
        if (self.index is None):
            self.load_embeddings()
        retriever = self.index.as_retriever(similarity_top_k=self.top_k)
        # Use llamaindex to find the most relevant documents for the query
        raw_results = retriever.retrieve(query)

        processor = SimilarityPostprocessor(similarity_cutoff=(self.min_relevance/100))
        results = processor.postprocess_nodes(raw_results)

        num = 0
        result=""
        for node in results:
            result +=f"Text({(num+1)}) = " + node.get_text() + f"\nScore({(num+1)})={node.get_score()}\n"
            num=num+1
        result = f"Number of chunks: {num}.\n" + result
        return result
        
    
#############################################################################################
#############################################################################################
#############################################################################################
class SentenceTransformerRetriever():
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


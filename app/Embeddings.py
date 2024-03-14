from abc import ABC, abstractmethod
from log import config, log_info, log_error

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import ServiceContext, SimpleDirectoryReader, Settings, VectorStoreIndex
from sentence_transformers import SentenceTransformer
import pickle

# Define the Embeddings Strategy Interface
class EmbeddingsStrategy(ABC):
    @abstractmethod
    def embeddings(self, query):
        pass

#############################################################################################
#############################################################################################
#############################################################################################
class Embeddings:
    def __init__(self, strategy: EmbeddingsStrategy):
        self.strategy = strategy

    def create_embeddings(self):
        # Placeholder for loading the corpus
        pass
    
#############################################################################################
#############################################################################################
#############################################################################################
class LlamaIndexEmbeddings(EmbeddingsStrategy):
    def __init__(self, index_path, data_folder, model_name, model_folder, chunk_size, chunk_overlap):
        self.index_path = index_path #-->env["embeddings_folder"]
        self.data_path = data_folder #-->env["data_folder"]
        self.model_name = model_name #-->env["embedding_model_name"]
        self.model_folder = model_folder #-->env["model_folder"]
        self.chunk_size = chunk_size #-->env["chunk_size"]
        self.chunk_overlap = chunk_overlap #-->env["chunk_overlap"]



    def create_embeddings(self):
        self._model = SentenceTransformer(self.model_name, cache_folder=self.model_folder) 
        self.text_splitter = SentenceSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        service_context_embedding = ServiceContext.from_defaults(llm=None, embed_model=self._model, transformations=[self.text_splitter])

        self.reader=SimpleDirectoryReader(self.data_path)
        documents = self.reader.load_data()
        log_info(f"Read {len(documents)} documents.")
        
        self.index =  VectorStoreIndex.from_documents(documents, service_context=service_context_embedding, show_progress=True)
        self.index.storage_context.persist(persist_dir=self.index_path)
        log_info("LlamaIndex embeddings created successfully.")

#############################################################################################
#############################################################################################
#############################################################################################
class SentenceTransformerEmbeddings(EmbeddingsStrategy):
    def __init__(self, model_name='all-MiniLM-L6-v2', filename='embeddings', top_k=5):
            log_info("Initializing SentenceTransformerRetrieval...")
            
            self.model = SentenceTransformer(model_name)
            
            log_info(f"Loaded SentenceTransformer model: {model_name}")

            self.corpus_embeddings = []
            self.corpus_texts

            self.filename = filename
            self.top_k = top_k



    def create_embeddings(self): #TODO: fer!
        pass



from abc import ABC
from log import config, logger

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import ServiceContext, SimpleDirectoryReader, Settings, VectorStoreIndex

from langchain_community.embeddings import HuggingFaceEmbeddings

import pickle

#############################################################################################
#############################################################################################
#############################################################################################
class Embeddings:
    def __init__(self):
        pass

    def create_embeddings(self):
        # Placeholder for loading the corpus
        pass
    
#############################################################################################
#############################################################################################
#############################################################################################
class LlamaIndexEmbeddings():
    def __init__(self, config, datafolder):
        self.index_path = config["vector_folder"] + datafolder
        self.data_path = config["data_folder"] + datafolder
        self.model_name = config["model_name"]
        self.model_folder = config["model_folder"]
        self.chunk_size = config["chunk_size"]
        self.chunk_overlap = config["chunk_overlap"]

    def create_embeddings(self):
        self._model = HuggingFaceEmbeddings(model_name=self.model_name, cache_folder=self.model_folder)
        self.text_splitter = SentenceSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        service_context_embedding = ServiceContext.from_defaults(llm=None, embed_model=self._model, transformations=[self.text_splitter])

        self.reader=SimpleDirectoryReader(self.data_path)
        documents = self.reader.load_data()
        logger.info(f"Read {len(documents)} documents.")
        
        self.index =  VectorStoreIndex.from_documents(documents, service_context=service_context_embedding, show_progress=True)
        self.index.storage_context.persist(persist_dir=self.index_path)
        logger.info("LlamaIndex embeddings created successfully.")

#############################################################################################
#############################################################################################
#############################################################################################
class SentenceTransformerEmbeddings():
    def __init__(self, config, datafolder):
            self.index_path = config["vector_folder"] + datafolder
            self.data_path = config["data_folder"] + datafolder
            self.model_name = config["model_name"]
            self.model_folder = config["model_folder"]
            self.chunk_size = config["chunk_size"]
            self.chunk_overlap = config["chunk_overlap"]

            # logger.info("Initializing SentenceTransformerRetrieval...")
            
            # self.model = SentenceTransformer(model_name)
            
            # logger.info(f"Loaded SentenceTransformer model: {model_name}")

            # self.corpus_embeddings = []
            # self.corpus_texts

            # self.filename = filename
            # self.top_k = top_k

    def create_embeddings(self): #TODO: fer!
        pass



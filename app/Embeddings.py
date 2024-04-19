import os
import glob
import pickle

from abc import ABC
from log import config, logger

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import ServiceContext, SimpleDirectoryReader, Settings, VectorStoreIndex

from langchain_community.embeddings import HuggingFaceEmbeddings

from sentence_transformers import SentenceTransformer, util

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
            self.top_k = config["number_of_documents"]

            logger.info("Initializing SentenceTransformerEmbbedings...")
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded SentenceTransformer model for embedding: {self.model_name}")

            self.corpus_embeddings = []
            self.corpus_texts = []

    def create_embeddings(self): #TODO: fer!
        # load documents
        self.load_documents()
        # Crear embeddings y añadirlos a corpus_embeddings
        self.encode_embeddings()
        # guardar la BD
        self.save_embeddings()

    def load_documents(self):
        for filepath in glob.glob(self.data_path):
            with open(filepath, 'r', encoding=config["data_encoding"]) as f:
                # Leemos todo el contenido del archivo en una sola cadena
                content = f.read()

                # Dividimos el contenido en párrafos
                paragraphs = content.split('\n\n')

                # Extendemos nuestra lista global de fragmentos de texto con estas oraciones
                self.corpus_texts.extend(paragraphs)
                
                logger.info(f"File {filepath} chunked.")

    def encode_embeddings(self):
        pass

    def save_embeddings(self):
        pass


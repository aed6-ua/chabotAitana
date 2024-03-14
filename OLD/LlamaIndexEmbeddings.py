from abc import ABC, abstractmethod
import logging

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import ServiceContext, SimpleDirectoryReader, Settings, VectorStoreIndex
from sentence_transformers import SentenceTransformer

from Retriever import RetrievalStrategy

from env import env

if (env["log"]=="Y"):
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class LlamaIndexEmbeddings:
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
        if (env["verbose"]=="Y"):
            print(f"Read {len(documents)} documents.")
        
        self.index =  VectorStoreIndex.from_documents(documents, service_context=service_context_embedding, show_progress=True)
        self.index.storage_context.persist(persist_dir=self.index_path)
        if (env["log"]=="Y"):
            logging.info("LlamaIndex embeddings created successfully.")


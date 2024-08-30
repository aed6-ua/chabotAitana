from pathlib import Path

#from log import config, logger
import log

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import ServiceContext, SimpleDirectoryReader, Settings, VectorStoreIndex

from langchain_community.embeddings import HuggingFaceEmbeddings

class LlamaIndexEmbeddings():
    def __init__(self, local_config, datafolder):
        self.index_path = local_config["vector_folder"] + datafolder
        self.data_path = local_config["data_folder"] + datafolder
        self.model_name = local_config["model_name"]
        self.model_folder = local_config["model_folder"]
        self.chunk_size = local_config["chunk_size"]
        self.chunk_overlap = local_config["chunk_overlap"]

    def create_embeddings(self):
        self._model = HuggingFaceEmbeddings(model_name=self.model_name, cache_folder=self.model_folder)
        self.text_splitter = SentenceSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        service_context_embedding = ServiceContext.from_defaults(llm=None, embed_model=self._model, transformations=[self.text_splitter])

        self.reader=SimpleDirectoryReader(self.data_path)
        documents = self.reader.load_data()
        log.logger.info(f"Read {len(documents)} documents.")
        
        self.index =  VectorStoreIndex.from_documents(documents, service_context=service_context_embedding, show_progress=True)
        self.index.storage_context.persist(persist_dir=self.index_path)
        log.logger.info("LlamaIndex embeddings created successfully.")


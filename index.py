import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr
from sentence_transformers import SentenceTransformer
import asyncio
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.indices.query.embedding_utils import get_top_k_embeddings
from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore
from typing import List, Any, Optional
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.core import (load_index_from_storage, StorageContext)



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

class IndexManager:
    def __init__(self, persist_dir="./storage"):
        self.text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=32)
        # Global settings
        Settings.embed_model = CustomEmbeddings()
        self.storage_context = StorageContext.from_defaults(persist_dir=persist_dir)

    def read(self, data_folder):
        reader = SimpleDirectoryReader(data_folder)
        return reader.load_data()
    
    def create_index(self, documents, index_id="es.aitana.index"):
        index = VectorStoreIndex.from_documents(documents, transformations=[self.text_splitter], show_progress=True)
        index.set_index_id(index_id)
        return index
    
    def index_from_directory(self, data_folder, index_id="es.aitana.index"):
        documents = self.read(data_folder)
        index = self.create_index(documents, index_id=index_id)
        index.storage_context.persist()
        return index
    
    def load_index(self, index_id="es.aitana.index"):
        return load_index_from_storage(self.storage_context, index_id=index_id)


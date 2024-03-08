import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader

class IngestionManager:
    def __init__(self, reader=SimpleDirectoryReader("./data")):
        self.reader = reader

    def read(self):
        return self.reader.load_data()

from llama_index.core import VectorStoreIndex
from llama_index.core import Settings

class VectorStoreManager:
    def __init__(self):
        self.text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=32)
        # Global settings
        Settings.embed_model = CustomEmbeddings()

    def create_index(self, documents):
        return VectorStoreIndex.from_documents(documents, transformations=[self.text_splitter], show_progress=True)
    


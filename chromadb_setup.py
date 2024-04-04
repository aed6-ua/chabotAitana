import chromadb

def initialize_chromadb_collection(collection_name="chatbot_documents"):
    client = chromadb.PersistentClient()
    collection = client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    return collection


from unstructured.ingest.connector.chroma import (
    ChromaAccessConfig,
    ChromaWriteConfig,
    SimpleChromaConfig,
)
from unstructured.ingest.runner.writers.chroma import ChromaWriter

def get_chroma_writer(collection_name="chatbot_documents", host="localhost", port=8000):
    # If collection does not exist, create it
    client = chromadb.HttpClient(host=host, port=port)
    client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    return ChromaWriter(
        connector_config=SimpleChromaConfig(
            access_config=ChromaAccessConfig(),
            host=host,
            port=port,
            collection_name=collection_name,
            tenant="default_tenant",
            database="default_database",
        ),
        write_config=ChromaWriteConfig(),
    )

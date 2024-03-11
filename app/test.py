# Test the IndexnManager with ./data

from IndexManager import IngestionManager, VectorStoreManager

ingestion_manager = IngestionManager()
documents = ingestion_manager.read()
print(f"Read {len(documents)} documents.")
vector_store_manager = VectorStoreManager()
index = vector_store_manager.create_index(documents)

# Save the index
index.storage_context.persist()
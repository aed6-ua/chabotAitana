# Test the IndexnManager with ./data

from index import IndexManager

index_manager = IndexManager()
index = index_manager.index_from_directory("./data")

# Test the LlamaIndexRetriever with the index
from retrieval import LlamaIndexRetriever

retriever = LlamaIndexRetriever(index)
results = retriever.retrieve("hola")
print(results)

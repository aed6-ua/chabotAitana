# Test the ChromaDB Retriever

from retrieval import ChromaDBRetriever
from model import EmbeddingsModel

# Load the embeddings model
embeddings_model = EmbeddingsModel("hackathon-pln-es/paraphrase-spanish-distilroberta")

# Create the retriever
retriever = ChromaDBRetriever(embeddings_model, collection_name="chatbot_documents")

# Test the retriever
query = "¿Que transporte hay para llegar a la UA?"
results = retriever.retrieve(query)

print(results)

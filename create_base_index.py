# Script to create the base index using documents from data folder and ingestion pipeline

# Import the necessary libraries
import os
import shutil

# Define the paths
data_folder = "data"
output_dir = "base_index"
collection_name = "base"

# Run the embedding pipeline
#run_embedding_pipeline(data_folder, output_dir, collection_name=collection_name)

# Load embeddings model
from model import EmbeddingsModel
from factory import load_config
config = load_config("config.json")
embeddings_model = EmbeddingsModel(config["retriever"]["llm_server"])


# Create folder for the assistant
shutil.rmtree(f"assistants/base", ignore_errors=True)
shutil.os.makedirs(f"assistants/base")
# Copy the files from data folder to the assistant folder
for file_name in os.listdir(data_folder):
    file_path = os.path.join(data_folder, file_name)
    shutil.copy(file_path, f"assistants/base/{file_name}")
from ingestion_pipeline import load_and_store, run_embedding_pipeline
run_embedding_pipeline(f"assistants/base", output_dir=f"assistants/base/index", collection_name="base")
load_and_store(f"assistants/base/index", embeddings_model=embeddings_model, collection_name="base")

# Print the success message
print("Base index created successfully!")

# Connect to the ChromaDB database
import chromadb
client = chromadb.HttpClient(host="localhost", port=8000)
# Peek at the collection
collection = client.get_collection(collection_name)
print(collection.peek())
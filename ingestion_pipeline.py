from unstructured.ingest.connector.local import SimpleLocalConfig
from unstructured.ingest.interfaces import (
    ChunkingConfig,
    PartitionConfig,
    ProcessorConfig,
    ReadConfig,
)
from unstructured.ingest.runner import LocalRunner
from chromadb_setup import get_chroma_writer
from model import EmbeddingsModel
import os
import json
import numpy

def run_embedding_pipeline(input_path, collection_name="chatbot_documents"):
    output_dir = f"local-output-to-{collection_name}"
    

    runner = LocalRunner(
        processor_config=ProcessorConfig(
            verbose=True,
            output_dir=output_dir,
            num_processes=2,
        ),
        connector_config=SimpleLocalConfig(input_path=input_path),
        read_config=ReadConfig(),
        partition_config=PartitionConfig(),
        chunking_config=ChunkingConfig(),#chunking_strategy="basic", max_characters=5000, new_after_n_chars=0),
        #writer=writer,
        #writer_kwargs={},
    )

    runner.run()

    

def load_and_store(output_dir, embeddings_model: EmbeddingsModel = None, collection_name="chatbot_documents"):
    writer = get_chroma_writer(collection_name=collection_name)
    json_files = []
    for file_name in os.listdir(output_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                json_data = json.load(file)
                json_files.append(json_data)

    #print("Loaded %d json files" % len(json_files))
    # Print first element of first file
    #print(json_files[0][0])

    writer = writer.get_connector()
    elements = []
    for json_data in json_files:
        for element in json_data:
            elements.append(writer.normalize_dict(element))
    
    #elements = elements[0:2]
        
    # Generate embeddings
    for element in elements:
        if embeddings_model is not None:
            element["embedding"] = embeddings_model.run(element["document"]).tolist()
        else:
            element["embedding"] = [None]
    
    # Write to ChromaDB
    
    #print(elements[0])
    #print("Writing %d elements to ChromaDB" % len(elements))
        
    writer.write_dict(elements_dict=elements)

if __name__ == '__main__':
    # Test
    model = EmbeddingsModel("hackathon-pln-es/paraphrase-spanish-distilroberta")
    load_and_store("local-output-to-chatbot_documents", model)
    
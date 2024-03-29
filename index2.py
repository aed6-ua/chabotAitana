from unstructured.partition.auto import partition
from model import EmbeddingsModel
from unstructured.documents.elements import NarrativeText
from unstructured.partition.text_type import sentence_count

def process_document(file_path, embeddings_model_path, min_sentence_count=2):
    # Initialize the embeddings model
    embeddings_model = EmbeddingsModel(embeddings_model_path)
    
    # Partition the document
    elements = partition(filename=file_path)
    
    # Filter elements: Here, we focus on NarrativeText with a sufficient sentence count
    filtered_elements = [el for el in elements if isinstance(el, NarrativeText) and sentence_count(el.text) >= min_sentence_count]
    
    # Generate embeddings for filtered elements
    embeddings = [embeddings_model.run(el.text) for el in filtered_elements]
    
    # Combine elements and their embeddings for further processing
    processed_elements = [{'text': el.text, 'embedding': emb} for el, emb in zip(filtered_elements, embeddings)]
    
    return processed_elements


def test():
    file_path = 'data/Doble_factor.es.txt'
    embeddings_model_path = "hackathon-pln-es/paraphrase-spanish-distilroberta"
    min_sentence_count = 2
    processed_elements = process_document(file_path, embeddings_model_path, min_sentence_count)
    # Save the processed elements to a file
    with open('processed_elements.txt', 'w') as file:
        for element in processed_elements:
            file.write(f"Text: {element['text']}\n")
            file.write(f"Embedding: {element['embedding']}\n")
            file.write("\n")


test()
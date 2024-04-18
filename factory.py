from assistant import Assistant
from model import LocalGenerationModel, TestModel, OpenAIGenerationModel, EmbeddingsModel
from retrieval import SimpleRetriever, LlamaIndexRetriever, ChromaDBRetriever
from LlamaindexAssistant import LlamaindexAssistant
from index import IndexManager
import json



# Factory function temporarily placed here
# Factory function to create an instance of the assistant
def create_assistant(config, retrieval_tool=None, generation_model=None, description="default", prompt_settings=None):
    config_assistant = config["assistant"]
    #memory_context_size = config["assistant"]["memory_context_size"]

    # Get model
    if generation_model is None:
        if config_assistant["test"]:
            generation_model = TestModel()
        else:
            generation_model = LocalGenerationModel(config_assistant["llm_server"])
            #generation_model = OpenAIGenerationModel(config_assistant["model_name"], config_assistant["openai_api_parameters"])
    
    #if prompt_settings is None:
    #    prompt_settings = config_assistant["prompt_settings"]

    return Assistant(model=generation_model, tools=[retrieval_tool], prompt_settings=prompt_settings, description=description)
    
# Factory function to create an instance of the retrieval tool
def create_retrieval_tool(config, embeddings_model=None, retrieval_type=None, collection_name=None):
    config_retrieval = config["retriever"]
    if retrieval_type is None:
        retrieval_type = config_retrieval["type"]
    elif embeddings_model is None and retrieval_type != "LlamaIndexRetriever":
        embeddings_model = EmbeddingsModel(config_retrieval["llm_server"])
    if collection_name is None:
        collection_name = config_retrieval["collection_name"]

    if retrieval_type == "SimpleRetriever":
        return SimpleRetriever(model=embeddings_model, filename=config_retrieval["filename"], top_k=config_retrieval["top_k"])
    elif retrieval_type == "LlamaIndexRetriever":
        return LlamaIndexRetriever(IndexManager(config_retrieval["index_path"]).load_index())
    elif retrieval_type == "ChromaDBRetriever":
        return ChromaDBRetriever(model=embeddings_model, collection_name=collection_name)
    else:
        raise ValueError(f"Unsupported retrieval type: {retrieval_type}")
    
# Load the configuration file
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config
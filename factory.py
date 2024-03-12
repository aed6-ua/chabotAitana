from GPTAssistant import GPTAssistant
from LocalAssistant import LocalAssistant
from Retriever import SentenceTransformerRetriever
from Retriever import LlamaIndexRetriever
from LlamaindexAssistant import LlamaindexAssistant
from Model import EmbeddingsModel


# Factory function temporarily placed here
# Factory function to create an instance of the assistant
def create_assistant(config, retrieval_tool=None, generation_model=None):
    config_assistant = config["assistant"]
    assistant_type = config_assistant["type"]
    #memory_context_size = config["assistant"]["memory_context_size"]

    if assistant_type == "GPTAssistant":
        return GPTAssistant(model=generation_model, retrieval_tool=retrieval_tool, prompt_settings=config_assistant["prompt_settings"])
    elif assistant_type == "LlamaindexAssistant":
        return LlamaindexAssistant(model_name=config_assistant["model_name"])
    elif assistant_type == "LocalAssistant":
        return LocalAssistant(model=generation_model, retrieval_tool=retrieval_tool, prompt_settings=config_assistant["prompt_settings"])
    else:
        raise ValueError(f"Unsupported assistant type: {assistant_type}")
    
# Factory function to create an instance of the retrieval tool
def create_retrieval_tool(config, embeddings_model=None):
    config_retrieval = config["retriever"]
    retrieval_type = config_retrieval["type"]

    if retrieval_type == "SentenceTransformerRetriever":
        return SentenceTransformerRetriever(model=embeddings_model, filename=config_retrieval["filename"], top_k=config_retrieval["top_k"])
    elif retrieval_type == "LlamaIndexRetriever":
        return LlamaIndexRetriever(index_path=config_retrieval["index_path"])
    else:
        raise ValueError(f"Unsupported retrieval type: {retrieval_type}")
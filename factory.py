from assistant import GPTAssistant, LocalAssistant, TestAssistant
from retrieval import SimpleRetriever, LlamaIndexRetriever, ChromaDBRetriever
from LlamaindexAssistant import LlamaindexAssistant
from index import IndexManager



# Factory function temporarily placed here
# Factory function to create an instance of the assistant
def create_assistant(config, retrieval_tool=None, generation_model=None, description="default"):
    config_assistant = config["assistant"]
    assistant_type = config_assistant["type"]
    #memory_context_size = config["assistant"]["memory_context_size"]

    if assistant_type == "GPTAssistant":
        return GPTAssistant(model=generation_model, tools=[retrieval_tool], prompt_settings=config_assistant["prompt_settings"], description=description)
    elif assistant_type == "LlamaindexAssistant":
        return LlamaindexAssistant(model_name=config_assistant["model_name"])
    elif assistant_type == "LocalAssistant":
        return LocalAssistant(model=generation_model, tools=[retrieval_tool], prompt_settings=config_assistant["prompt_settings"], description=description)
    elif assistant_type == "TestAssistant":
        return TestAssistant(model=generation_model, tools=[retrieval_tool], prompt_settings=config_assistant["prompt_settings"], description=description)
    else:
        raise ValueError(f"Unsupported assistant type: {assistant_type}")
    
# Factory function to create an instance of the retrieval tool
def create_retrieval_tool(config, embeddings_model=None):
    config_retrieval = config["retriever"]
    retrieval_type = config_retrieval["type"]

    if retrieval_type == "SimpleRetriever":
        return SimpleRetriever(model=embeddings_model, filename=config_retrieval["filename"], top_k=config_retrieval["top_k"])
    elif retrieval_type == "LlamaIndexRetriever":
        return LlamaIndexRetriever(IndexManager(config_retrieval["index_path"]).load_index())
    elif retrieval_type == "ChromaDBRetriever":
        return ChromaDBRetriever(model=embeddings_model, collection_name=config_retrieval["collection_name"])
    else:
        raise ValueError(f"Unsupported retrieval type: {retrieval_type}")
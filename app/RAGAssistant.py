from abc import ABC, abstractmethod
from log import config, logger

from GPTLLM import GPTLLM
from LlamaIndexLLM import LlamaIndexLLM
from LocalLLM import LocalLLM

from LlamaIndexRetriever import LlamaIndexRetriever
from SentenceTransformerRetriever import SentenceTransformerRetriever

class RAGAssistant():
    def __init__(self, local_config): #model_name="gpt-3.5-turbo", retrieval_tool=None, prompt_settings=None, api_parameters=None):
        # If està configurat, creem les eines accessòries:
        self.index_path = local_config["vector_folder"] + local_config["RAG_DB_Folder"]
        self.retriever=None
        self.LLM=None
        self.error_msg = local_config["error_msg"]

        self.prompt_settings = local_config["prompt_settings"]

        if local_config["retriever"]!='':
            self.createRetriever(local_config["retriever"], local_config["RAG_DB_Folder"])

        if local_config["LLM"]!='':
            self.createLLM(local_config["LLM"])
        

    def createRetriever(self, model, datafolder):
        if model=="SentenceTransformerRetriever":
            self.retriever = SentenceTransformerRetriever(config["SentenceTransformerRetriever"], datafolder)
            self.retriever.load_embeddings()
        elif model=="LlamaIndexRetriever":
            self.retriever = LlamaIndexRetriever(config["LlamaIndexRetriever"], datafolder)
            self.retriever.load_embeddings()
        else:
            logger.error(f"Retriever model {model} not suported")

    def createLLM(self, model):
        if model=="GPTLLM":
            self.LLM = GPTLLM(config["GPTLLM"], self.prompt_settings)
        elif model=="LlamaIndexLLM":
            self.LLM = LlamaIndexLLM(config["LlamaIndexLLM"], self.prompt_settings)
        elif model=="LocalLLM":
            self.LLM = LocalLLM(config["LocalLLM"], self.prompt_settings)
        else:
            logger.error(f"Retriever model {model} not suported")

    def setRetriever(self, retriever):
        self.retriever = retriever

    def setLLM(self, LLM):
        self.LLM = LLM

    def prompt(self, message, history):
        response=''
        error=False
        try:
            if self.retriever!=None:
                RAG_context = self.retriever.retrieve(message)
                
                if self.LLM!=None: 
                    response = self.LLM.process_message(message, RAG_context, history)
                    if (response==config["global"]["token_error"]):
                        error=True
                else:
                    logger.error("LLM not existent")
                    error=True
            else:
                logger.error("Retriever not existent")
                error=True
        except Exception as e:
             logger.error(f"Error processing message: {e}")
             error=True

        if error==True:
            response = self.error_msg

        return response
    
    # def _use_retrieval_tool_if_available(self, message, context):
    #     if self.tools and self.tools[0]:
    #         try:
    #             # List of tuples with the retrieved passages and their scores
    #             retrieval_result = self.tools[0].retrieve(message)
    #             # Convert to string with each result between triple quotes
    #             return "\n\n".join([f'"""{result}"""' for result, _ in retrieval_result])
    #         except Exception as e:
    #             logger.error(f"Retrieval tool failed: {e}")
    #     return context

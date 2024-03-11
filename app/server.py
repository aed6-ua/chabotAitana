import logging
from fastapi import Body, FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from uuid import uuid4

class MessageSchema(BaseModel):
    message: str

# Import classes here
from Conversation import Conversation
from GPTAssistant import GPTAssistant
from Retriever import SentenceTransformerRetriever
from Retriever import LlamaIndexRetriever
from LlamaindexAssistant import LlamaindexAssistant

# Configure logging
logging.basicConfig(level=logging.INFO, filename='app.log', filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import json

# Load the configuration file
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

# Accessing the configuration
config = load_config("config.json")

app = FastAPI()

# In-memory storage for conversations
conversations = {}

# Factory function temporarily placed here
# Factory function to create an instance of the assistant
def create_assistant(config, retrieval_tool=None):
    config_assistant = config["assistant"]
    assistant_type = config_assistant["type"]
    #memory_context_size = config["assistant"]["memory_context_size"]

    if assistant_type == "GPTAssistant":
        return GPTAssistant(model_name=config_assistant["model_name"], retrieval_tool=retrieval_tool, prompt_settings=config_assistant["prompt_settings"])
    elif assistant_type == "LlamaindexAssistant":
        return LlamaindexAssistant(model_name=config_assistant["model_name"])
    else:
        raise ValueError(f"Unsupported assistant type: {assistant_type}")
    
# Factory function to create an instance of the retrieval tool
def create_retrieval_tool(config):
    config_retrieval = config["retriever"]
    retrieval_type = config_retrieval["type"]

    if retrieval_type == "SentenceTransformerRetriever":
        return SentenceTransformerRetriever(model_name=config_retrieval["model_name"], filename=config_retrieval["filename"], top_k=config_retrieval["top_k"])
    elif retrieval_type == "LlamaIndexRetriever":
        return LlamaIndexRetriever(index_path=config_retrieval["index_path"])
    else:
        raise ValueError(f"Unsupported retrieval type: {retrieval_type}")
    
# Create an instance of the assistant
assistant = create_assistant(config)

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    logger.error(f"HTTP error occurred: {exc}")
    return JSONResponse(status_code=exc.status_code, content={"message": str(exc.detail)})

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    logger.error(f"Validation error: {exc}")
    return JSONResponse(status_code=400, content={"message": "Validation Error"})

@app.post("/start")
async def start_conversation():
    conversation_id = str(uuid4())
    conversations[conversation_id] = Conversation(conversation_id)
    return {"conversation_id": conversation_id}

@app.post("/send/{conversation_id}")
async def send_message(conversation_id: str, message_body: MessageSchema):
    message = message_body.message
    conversation = conversations.get(conversation_id)
    if not conversation:
        raise StarletteHTTPException(status_code=404, detail="Conversation not found")

    # Process the message with the assistant
    context = conversation.get_context()
    response = assistant.process_message(message, context)
    conversation.add_message("user", message)
    conversation.add_message("assistant", response)
    
    return {"response": response}

def store_conversation(conversation_id):
    conversation = conversations[conversation_id]
    serialized_conversation = str(conversation)
    with open(f"conversations/{conversation_id}.log", "w") as file:
        file.write(serialized_conversation)
    logger.info(f"Conversation {conversation_id} stored")

@app.post("/end/{conversation_id}")
async def end_conversation(conversation_id: str):
    if conversation_id not in conversations:
        raise StarletteHTTPException(status_code=404, detail="Conversation not found")
    
    store_conversation(conversation_id)
    del conversations[conversation_id]  # Optionally remove from active conversations
    return {"message": "Conversation ended and stored"}

# Hello world route
@app.get("/")
async def read_root():
    return {"Hello": "World"}

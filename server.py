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
from factory import create_assistant, create_retrieval_tool

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

# Load models
from Model import OpenAIGenerationModel
from Model import EmbeddingsModel
embeddings_model = EmbeddingsModel(config["retriever"]["model_name"])
generation_model = OpenAIGenerationModel(config["assistant"]["model_name"])

app = FastAPI()

# In-memory storage for conversations
conversations = {}

# Dictionary to store the assistants with their IDs
assistants = {}

# Create an instance of the assistant
base_assistant = create_assistant(config, generation_model=generation_model, retrieval_tool=create_retrieval_tool(config, embeddings_model=embeddings_model))
assistants["base"] = base_assistant

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    logger.error(f"HTTP error occurred: {exc}")
    return JSONResponse(status_code=exc.status_code, content={"message": str(exc.detail)})

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    logger.error(f"Validation error: {exc}")
    return JSONResponse(status_code=400, content={"message": "Validation Error"})

# List all assistants
@app.get("/assistants")
async def list_assistants():
    return {"assistants": list(assistants.keys())}

# Create new assistant
@app.post("/create_assistant")
async def create_assistant_endpoint():
    return {"message": "Not implemented"}

@app.post("/start")
async def start_conversation():
    conversation_id = str(uuid4())
    conversations[conversation_id] = Conversation(conversation_id)
    return {"conversation_id": conversation_id}

@app.post("/send/{conversation_id}/{assistant_id}")
@app.post("/send/{conversation_id}")
async def send_message(conversation_id: str, message_body: MessageSchema, assistant_id=None):
    message = message_body.message
    conversation = conversations.get(conversation_id)
    if not conversation:
        raise StarletteHTTPException(status_code=404, detail="Conversation not found")

    # Process the message with the assistant
    context = conversation.get_context()
    if assistant_id is None:
        assistant_id = "base"
    response = assistants[assistant_id].process_message(message, context)
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

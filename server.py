import logging
import os
import pickle
from fastapi import Body, FastAPI, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from uuid import uuid4
import shutil

class MessageSchema(BaseModel):
    message: str
    top_k: int
    max_tokens: int
    temperature: float

# Import classes here
from conversation import Conversation
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
from model import OpenAIGenerationModel, EmbeddingsModel, LocalGenerationModel
try:
    embeddings_model = EmbeddingsModel(config["retriever"]["llm_server"])
    generation_model = LocalGenerationModel(config["assistant"]["llm_server"])
        #generation_model = OpenAIGenerationModel(config["assistant"]["model_name"])
except Exception as e:
    logger.error(f"Error loading models: {e}")

app = FastAPI()

# In-memory storage for conversations
conversations = {}

# Dictionary to store the assistants with their IDs
assistants = {}

# Load stored assistants
if os.path.exists("assistants") and os.listdir("assistants"):
    for file in os.listdir("assistants"):
        assistant_id = file.split(".")[0]
        if file.endswith(".pkl"):
            with open(f"assistants/{assistant_id}.pkl", "rb") as file:
                loaded_config = pickle.load(file)
                assistants[assistant_id] = create_assistant(loaded_config, retrieval_tool=create_retrieval_tool(loaded_config, embeddings_model=embeddings_model), generation_model=generation_model, description=loaded_config["assistant"]["description"])
                logger.info(f"Loaded assistant {assistant_id}")
else:
    logger.error("No assistants found in the 'assistants' directory.")
        

# Create an instance of the assistant
base_assistant = create_assistant(config, generation_model=generation_model, retrieval_tool=create_retrieval_tool(config, embeddings_model=embeddings_model), description="Base assistant")
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
    assistants_list = []
    for assistant_id, assistant in assistants.items():
        assistants_list.append({"assistant_id": assistant_id, "description": assistant.description})
    return assistants_list

@app.get("/create_assistant")
async def main():
    content = """
<body>
<form action="/create_assistant" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input name="description" type="text" placeholder="Enter description">
<input type="submit">
</form>
</body>
    """
    return HTMLResponse(content=content)

# Create new assistant
@app.post("/create_assistant")
async def create_assistant_endpoint(files: list[UploadFile], description: str = Body(...)):
    assistant_id = str(uuid4())
    # Create folder for the assistant
    shutil.rmtree(f"assistants/{assistant_id}", ignore_errors=True)
    shutil.os.makedirs(f"assistants/{assistant_id}")
    if files:
        for file in files:
            path = f"assistants/{assistant_id}/{file.filename}"
            with open(path, "w+b") as buffer:
                shutil.copyfileobj(file.file, buffer)
    from ingestion_pipeline import load_and_store, run_embedding_pipeline
    run_embedding_pipeline(f"assistants/{assistant_id}", output_dir=f"assistants/{assistant_id}/local-output", collection_name=assistant_id)
    load_and_store(f"assistants/{assistant_id}/local-output", embeddings_model=embeddings_model, collection_name=assistant_id)
    retriever_config = {
            "retriever": {
            "type": "ChromaDBRetriever",
            "model_name": "hackathon-pln-es/paraphrase-spanish-distilroberta",
            "top_k": 5,
            "collection_name": assistant_id
        }
    }
    assistant = create_assistant(config, generation_model=generation_model, retrieval_tool=create_retrieval_tool(retriever_config, embeddings_model=embeddings_model), description=description)
    assistants[assistant_id] = assistant
    
    return {"assistant_id": assistant_id}


@app.post("/start")
async def start_conversation():
    conversation_id = str(uuid4())
    conversations[conversation_id] = Conversation(conversation_id)
    return {"conversation_id": conversation_id}

@app.post("/send/{conversation_id}/{assistant_id}")
@app.post("/send/{conversation_id}")
async def send_message(conversation_id: str, message_body: MessageSchema, assistant_id=None):
    message = message_body.message
    top_k = message_body.top_k
    max_tokens = message_body.max_tokens
    temperature = message_body.temperature
    conversation = conversations.get(conversation_id)
    if not conversation:
        raise StarletteHTTPException(status_code=404, detail="Conversation not found")

    # Process the message with the assistant
    context = conversation.get_context()
    if assistant_id is None:
        assistant_id = "base"
    logger.info(f"Processing message with assistant {assistant_id}")
    response, message_context = assistants[assistant_id].process_message(message, context, top_k=top_k, max_tokens=max_tokens, temperature=temperature)
    conversation.add_message("user", message_context)
    conversation.add_message("assistant", response)
    
    return {"response": response}

def store_conversation(conversation_id, stars):
    conversation = conversations[conversation_id]
    conversation.stars(stars)
    serialized_conversation = str(conversation)
    with open(f"conversations/{conversation_id}.log", "w") as file:
        file.write(serialized_conversation)
    logger.info(f"Conversation {conversation_id} stored")

class StarsSchema(BaseModel):
    stars: int

@app.post("/end/{conversation_id}")
async def end_conversation(conversation_id: str, stars: StarsSchema | None = None):
    if stars is not None:
        stars = stars.stars
    else:
        stars = -1
    if conversation_id not in conversations:
        raise StarletteHTTPException(status_code=404, detail="Conversation not found")
    
    store_conversation(conversation_id, stars)
    del conversations[conversation_id]  # Optionally remove from active conversations
    return {"message": "Conversation ended and stored"}

# Hello world route
@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.on_event("shutdown")
def shutdown_event():
    logger.info("Shutting down...")
    # Store all assistants as JSON config files
    for assistant_id, assistant in assistants.items():
        with open(f"assistants/{assistant_id}.pkl", "wb") as file:
            pickle.dump(assistant.get_config(config), file)
    logger.info("Stored all assistants")


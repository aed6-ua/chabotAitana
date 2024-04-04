import logging
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
from model import OpenAIGenerationModel, EmbeddingsModel
embeddings_model = EmbeddingsModel(config["retriever"]["model_name"])
generation_model = OpenAIGenerationModel(config["assistant"]["model_name"])

app = FastAPI()

# In-memory storage for conversations
conversations = {}

# Dictionary to store the assistants with their IDs
assistants = {}

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
    conversation = conversations.get(conversation_id)
    if not conversation:
        raise StarletteHTTPException(status_code=404, detail="Conversation not found")

    # Process the message with the assistant
    context = conversation.get_context()
    if assistant_id is None:
        assistant_id = "base"
    logger.info(f"Processing message with assistant {assistant_id}")
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

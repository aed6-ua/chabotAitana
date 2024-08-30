from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any

#from log import config, logger
import log

class ChatServer:
    def __init__(self, assistant):
        self.app = FastAPI()
        self.setup_cors()
        self.setup_routes()
        self.assistant = assistant

    def setup_cors(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def setup_routes(self):
        @self.app.on_event("shutdown")
        def shutdown_event():
            log.logger.info("Server shutting down...")
            # TODO: Guardar las sesiones y el historial de chat

        @self.app.get("/")
        async def root():
            response = {
                "assistant": self.assistant.assistant_name,
                "retriever":log.config[self.assistant.assistant_name]["retriever"],
                "LLM": log.config[self.assistant.assistant_name]["LLM"],
                "context": log.config[self.assistant.assistant_name]["RAG_DB_Folder"]
            }
            return response
        
        @self.app.post("/chat")
        async def chat(request_data: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
            query = request_data.get("query")
            session_id = request_data.get("session")
            #TODO: conversation (history)
            history=[]
            result = self.assistant.prompt(query,history)
            response = {
                "session_id": session_id,
                "text": result
            }
            return response
        """
        @self.app.post("/changeChatMode")
        async def change_chat_mode(request_data: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
            mode = request_data.get("mode")
            if mode != "qa" and mode != "chat":
                return {"status": "error", "message": f"Invalid mode: {mode}"}
            self.chatgpt.set_qa_mode(mode == mode)
            return {"status": "success", "message": f"Mode changed to {mode}"}
        
        @self.app.post("/changeTemperature")
        async def change_temperature(request_data: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
            temperature = request_data.get("temperature")
            self.chat_processor.set_chatgpt_temperature(temperature)
            return {"status": "success", "message": f"Temperature changed to {temperature}"}
        """
        @self.app.on_event("startup")
        async def startup_event():
            log.logger.info("Server starting up...")

    def run(self):
        import uvicorn
        ip = log.config['global']["ip"]
        port = log.config['global']["port"]
        uvicorn.run(self.app, host=ip, port=int(port))

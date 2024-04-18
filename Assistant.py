import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Open intro.txt file
with open("intro.txt", "r") as file:
    introduction = file.read()


class Assistant:
    def __init__(self, model, tools=None, prompt_settings=None, description="default"):
        self.tools = tools or []
        self.model = model
        self.description = description
        self.prompt_settings = {
            "language": "es",
            "introduction": introduction,
            "context": ""
        }

    def process_message(self, message, context, top_k=1, max_tokens=150, temperature=0.7):
        """
        Processes a received message using the provided context.
        
        :param message: The message to process.
        :param context: The context of the conversation.
        :return: The response message.
        """
        try:
            enhanced_context = self._use_retrieval_tool_if_available(message, context, top_k=top_k)
            message = f"Contexto:\n{enhanced_context}\n\nPregunta: {message}\n\nRespuesta: "
            messages = [
                {"role": "system", "content": self.prompt_settings["introduction"]},
                {"role": "user", "content": message},
            ]
            try:
                # Debug print
                #print(messages)
                return self.model.run(messages, max_tokens, temperature), message
            except Exception as e:
                logging.error(f"Failed to generate response: {e}")
                # Instead of just raising the exception, we handle it gracefully
                return "I'm sorry, I encountered an error trying to generate a response. Please try again later.", context
        except Exception as e:
            logging.error(f"Error processing message: {e}")
            return "I'm sorry, I encountered an error processing your request.", context

    def _use_retrieval_tool_if_available(self, message, context, top_k=1):
        if self.tools and self.tools[0]:
            try:
                # List of tuples with the retrieved passages and their scores
                retrieval_result = self.tools[0].retrieve(message, top_k=top_k)
                logging.info(f"Number of chunks retrieved: {len(retrieval_result)}")
                text = '"""'
                for passage in retrieval_result:
                    text = text + f"{passage}\n\n"
                    logging.info(f"Retrieved passage: {len(passage)}\n\n")
                text = text + '"""\n\n'
                return text
            except Exception as e:
                logging.warning(f"Retrieval tool failed: {e}")
        return context
    
    def get_config(self, config):
        from model import LocalGenerationModel, TestModel
        return {
            "assistant": {
                "test": True if isinstance(self.model, TestModel) else False,
                "local": True if isinstance(self.model, LocalGenerationModel) else False,
                "llm_server": config["assistant"]["llm_server"],
                "model_name": config["assistant"]["model_name"],
                "memory_context_size": config["assistant"]["memory_context_size"],
                "prompt_settings": self.prompt_settings,
                "openai_api_parameters": config["assistant"]["openai_api_parameters"],
                "description": self.description
            },
            
            "retriever": self.tools[0].get_config(config) if self.tools and self.tools[0] else {}
        }
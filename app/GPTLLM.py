import os
from log import config, logger

from openai import OpenAI

class GPTLLM():
    def __init__(self, local_config, prompt_settings):
        
        self._set_API_KEY()
        
        self.client = OpenAI()
        self.model_name = local_config["model_name"]
        self.prompt_settings = prompt_settings
        self.api_parameters = {
            "temperature": local_config["temperature"],
            "max_tokens": local_config["max_tokens"]
        }

    def _set_API_KEY(self):
        key = 'OPENAI_API_KEY'
        aux = os.getenv(key)
        if aux==None or aux=='':
            with open('apikey.txt', 'r') as file:
                value = file.read().strip()
            os.environ[key] = value

    def process_message(self, message, context, history):
        try:
            response = self._generate_response(message, context, history)
            return response
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return config["global"]["token_error"]

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
    

    def _generate_response(self, message, context, history):
        message = f"Contexto:\n{context}\n\Pregunta:\n{message}"
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.prompt_settings["introduction"]},
                    {"role": "user", "content": message},
                ],
                model=self.model_name,
                temperature=self.api_parameters["temperature"],
                max_tokens=self.api_parameters["max_tokens"]
            )
            response = chat_completion.choices[0].message.content
            return response.strip()
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            # Instead of just raising the exception, we handle it gracefully
            return "I'm sorry, I encountered an error trying to generate a response. Please try again later."
 
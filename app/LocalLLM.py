import os
import requests
from log import config, logger



class LocalLLM():
    def __init__(self, local_config, prompt_settings):
        self.model_name = local_config["model_name"]
        self.prompt_settings = prompt_settings

        self.url = local_config["server_url"]
        self.port = local_config["server_port"]
        self.endPoint = local_config["server_endPoint"]

        self.api_parameters = {
            "temperature": local_config["temperature"],
            "max_tokens": local_config["max_tokens"],
            "stream": local_config["stream"]
        }

    def process_message(self, message, context, history):
        try:
            response = self._generate_response(message, context, history)
            return response
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return config["global"]["token_error"]
        
    def _generate_response(self, message, context, history):
        url = self.url
        if self.port!=None and self.port!="":
            url = url + ":" + str(self.port)
        url = url + self.endPoint
        print(url)
        message = f"Contexto:\n{context}\n\Pregunta:\n{message}"

        headers={}

        body = {
            "model": self.model_name,
            "stream": self.api_parameters["stream"],
            "messages": [
                    {"role": "system", "content": self.prompt_settings["introduction"]},
                    {"role": "user", "content": message},
                ],
            "options": {
                "temperature": self.api_parameters["temperature"],
                "num_predict": self.api_parameters["max_tokens"]
            }
        }

        try:
            # Make the POST request
            response = requests.post(url, json=body, headers=headers)
        
            # Check if the request was successful
            response.raise_for_status()
        
            # Print the successful response
            print("Response Status Code:", response.status_code)
            print("Response Content:", response.json())
            
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"Failed to generate response, HTTP Error: {http_err}. Response: {response.text}")
            return "I'm sorry, I encountered an error trying to generate a response. Please try again later."
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            # Instead of just raising the exception, we handle it gracefully
            return "I'm sorry, I encountered an error trying to generate a response. Please try again later."

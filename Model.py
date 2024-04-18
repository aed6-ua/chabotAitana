from abc import ABC, abstractmethod
import requests

class Model(ABC):

    @abstractmethod
    def run(self, input):
        pass
    
    @abstractmethod
    def get_config(self):
        """
        Get a JSON representation of the model configuration for storage. The model can be reconstructed using the configuration.
        """
        pass


class EmbeddingsModel(Model):
    def __init__(self, server_url):
        self.server = server_url


    def run(self, input):
        print("Sending: ", input)
        response = requests.post(f"{self.server}/embed", params={"input": input})
        return response.json()["embeddings"]
    
    def get_config(self):
        return {"server_url": self.server}


class LocalGenerationModel(Model):
    def __init__(self, server_url):
        self.server = server_url

    def run(self, input, max_tokens=150, temperature=0.7):
        # The input is a list of messages like:
        # [
        #     {"role": "system", "content": self.prompt_settings["introduction"]},
        #     {"role": "user", "content": message},
        # ]
        input = {"messages": input,
                 "max_tokens": max_tokens,
                 "temperature": temperature}
        response = requests.post(f"{self.server}/generate", json=input)
        return response.json()["answer"]
    
    def get_config(self):
        return {"server_url": self.server}

class OpenAIGenerationModel(Model):
    def __init__(self, model_name="gpt-3.5-turbo", api_parameters=None):
        print(f"Initializing OpenAI generation model with model_name: {model_name}")
        from openai import OpenAI
        self.client = OpenAI()
        self.model_name = model_name
        self.api_parameters = api_parameters if api_parameters is not None else {
            "temperature": 0.7,
            "max_tokens": 1500
        }

    def run(self, messages, max_tokens=150, temperature=0.7):
        print("Sending messages to OpenAI:", messages) # Debug print
        chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature
            )
        response = chat_completion.choices[0].message.content
        return response
    
    def get_config(self):
        return {"model_name": self.model_name, "api_parameters": self.api_parameters}

class TestModel(Model):
    def __init__(self):
        print("Initializing test model")

    def run(self, input):
        return f"Test response to input: {input}"
    
    def get_config(self):
        return {}
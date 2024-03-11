from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class Model(ABC):

    @abstractmethod
    def load_model(self, model_path):
        pass

    @abstractmethod
    def run(self, input):
        pass


class EmbeddingsModel(Model):
    def __init__(self, model_path):
        print(f"Initializing embeddings model with model_path: {model_path}")
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        print(f"Loading embeddings model from {model_path}")
        return SentenceTransformer(model_path)

    def run(self, input):
        return self.model.encode(input, convert_to_tensor=True)

class LocalGenerationModel(Model):
    def __init__(self, model_path):
        print(f"Initializing local generation model with model_path: {model_path}")
        self.tokenizer, self.model = self.load_model(model_path)

    def load_model(self, model_path):
        print(f"Loading local generation model from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        return tokenizer, model

    def run(self, input):
        prompt = self.tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        output = self.model.generate(input_ids=inputs.to(self.model.device), max_new_tokens=150)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

class OpenAIGenerationModel(Model):
    def __init__(self, model_name="gpt-3.5-turbo", api_parameters=None):
        print(f"Initializing OpenAI generation model with model_name: {model_name}")
        from openai import OpenAI
        self.client = OpenAI()
        self.model_name = model_name
        self.api_parameters = api_parameters if api_parameters is not None else {
            "temperature": 0.7,
            "max_tokens": 150
        }

    def load_model(self, model_name):
        # No need to load the model, it's already loaded
        pass

    def run(self, messages):
        chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
            )
        response = chat_completion.choices[0].message.content
        return response
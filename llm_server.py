import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

# The input is a list of messages like:
# [
#     {"role": "system", "content": self.prompt_settings["introduction"]},
#     {"role": "user", "content": message},
# ]
class Input(BaseModel):
    messages: list
    max_tokens: int
    temperature: float

# Load the configuration file
print("Loading configuration file")
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

# Accessing the configuration
print("Accessing the configuration")
config = load_config("config.json")

print("Loading models")
if config["retriever"]["local"] == True:
    print("Loading embeddings model")
    embeddings_model = SentenceTransformer(config["retriever"]["model_name"])
if config["assistant"]["local"] == True:
    # "LenguajeNaturalAI/leniachat-gemma-2b-v0"
    # import torch
    # print("Loading generation model")
    # generation_tokenizer = AutoTokenizer.from_pretrained(config["assistant"]["model_name"], trust_remote_code=True)
    # generation_model = AutoModelForCausalLM.from_pretrained(config["assistant"]["model_name"], trust_remote_code=True)
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
else:
    from openai import OpenAI
    client = OpenAI()
if config["retriever"]["rerank"] == True:
    print("Loading reranker model")
    reranker_model = CrossEncoder(config["retriever"]["reranker_model_name"])

def get_rank(query, candidates):
    # Create input
    cross_input = [[query, candidate] for candidate in candidates]
    cross_scores = reranker_model.predict(cross_input)
    hits = [{"candidate": candidate, "cross-score": cross_score} for candidate, cross_score in zip(candidates, cross_scores)]
    for idx in range(len(cross_scores)):
        hits[idx]['cross-score'] = cross_scores[idx]
    hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
    return hits

def get_embeddings(input):
    #print("Processing input:", input)
    return embeddings_model.encode(input, convert_to_numpy=True)

def generate_response(input, max_tokens=150, temperature=0.7):
    if config["assistant"]["local"] == True:
        # inputs = generation_tokenizer.apply_chat_template(input, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        # #inputs = generation_tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        # prompt_length = inputs.shape[1]
        # with torch.no_grad():
        #     output = generation_model.generate(inputs, max_new_tokens=max_tokens)
        # response = generation_tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)
        print("Sending messages to LMStudio with parameters max_tokens:", max_tokens, "temperature:", temperature) # Debug print 
        chat_completion = client.chat.completions.create(
                messages=input,
                model=config["assistant"]["model_name"],
                temperature=temperature,
                max_tokens=max_tokens
            )
        response = chat_completion.choices[0].message.content
    else:
        
        print("Sending messages to OpenAI with parameters max_tokens:", max_tokens, "temperature:", temperature) # Debug print 
        chat_completion = client.chat.completions.create(
                messages=input,
                model=config["assistant"]["model_name"],
                temperature=temperature,
                max_tokens=max_tokens
            )
        response = chat_completion.choices[0].message.content
    return response

app = FastAPI()


@app.post("/embed")
def embed(input: str):
    #print("Processing input:", input)
    embeddings = get_embeddings(input).tolist()
    return {"embeddings": embeddings}

@app.post("/generate")
def generate(input: Input):
    messages = input.messages
    max_tokens = input.max_tokens
    temperature = input.temperature
    #print("Processing input:", input)
    answer = generate_response(messages, max_tokens=max_tokens, temperature=temperature)
    return {"answer": answer}

@app.post("/rank")
def rank(input: Input):
    #print("Processing input:", input)
    candidates = input.messages[1:]
    query = input.messages[0]
    hits = get_rank(query, candidates)
    return hits


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8001)



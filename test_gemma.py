from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "LenguajeNaturalAI/leniachat-gemma-2b-v0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Generar texto
with open("intro.txt", "r") as file:
    introduction = file.read()
messages = [
  {"role": "system", "content": introduction},
  {"role": "user", "content": "hola"}
]
input_ids = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt")
with torch.no_grad():
  output = model.generate(input_ids, max_new_tokens=50)
response = tokenizer.decode(output[0], skip_special_tokens=True)
print(response)
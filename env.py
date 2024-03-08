#Global values:
env = {
    "embedding_model_name": "hackathon-pln-es/paraphrase-spanish-distilroberta",
    "model_folder": "model/",
    "vector_folder": "store/",
    "num_sim_chunks": 5,
    "chunk_size": 1000,
    "%_sim_relevance": 40,
    "data_folder": "data/",
    "model_mode": "llamaindex",
    "working_mode": "test",
    "model_temperature": 0.01,
    "url_openai_chat": "https://api.openai.com/v1/chat/completions",
    "apikey_openai": "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",
    "openai_model": "gpt-3.5-turbo",
    "openai_max_tokens": 500,
    "serverURL": "172.25.2.137",
    "serverPort": 8000,
    "log": "Y",
    "log_folder": "log/",
    "verbose": "YA", #'': mute, 'Y': normal verbose, 'A': tell me everything
    "context_file_es": "texts/generalcontext.es.txt",
    "usermessage_file": "texts/usermessage.txt",
}

special_tokens = {
    "model": "<model>",
    "error": "<error>",
    "fall_back": "<fall_back>",
    "llm": "<llm>"
}
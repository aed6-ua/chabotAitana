from log import config, logger
from Embeddings import LlamaIndexEmbeddings, SentenceTransformerEmbeddings
from Retriever import LlamaIndexRetriever, SentenceTransformerRetriever

def do_embeddings(datafolder):
    error=False
    model = config["global"]["embeddings"]
    logger.info(f"Working creating Embeddings (with {model}) from data folder: {datafolder}")
    if model=="SentenceTransformerEmbeddings":
        embeder = SentenceTransformerEmbeddings(config["SentenceTransformerEmbeddings"], datafolder)
    elif model=="LlamaIndexEmbeddings":
        embeder = LlamaIndexEmbeddings(config["LlamaindexEmbeddings"], datafolder)
    else:
        logger.error(f"Embeddings model {model} not suported")
        error=True
    if not error:
        embeder.create_embeddings()

def do_retrieve(datafolder, prompt_filename=''):
    error=False
    model = config["global"]["retriever"]
    logger.info(f"Working as retriever (with {model}) from data folder: {datafolder}")
    if model=="SentenceTransformerRetriever":
        retriever = SentenceTransformerRetriever(config["SentenceTransformerRetriever"], datafolder)
    elif model=="LlamaIndexRetriever":
        retriever = LlamaIndexRetriever(config["LlamaIndexRetriever"], datafolder)
    else:
        logger.error(f"Retriever model {model} not suported")
        error=True

    if not error:
        if prompt_filename!='':
            with open(prompt_filename) as file:
                for line in file:
                    retriever.retrieve(line)
        else:
            prompt = input("Prompt: ")
            retriever.retrieve(prompt)

def do_assistant(datafolder):
    logger.info(f"Working as Asistant from RAG context: {datafolder}")
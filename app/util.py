from log import config, logger
from Embeddings import LlamaIndexEmbeddings, SentenceTransformerEmbeddings

def do_embeddings(datafolder):
    model = config["global"]["embeddings"]
    logger.info(f"Working creating Embeddings (with {model}) from data folder: {datafolder}")
    if model=="SentenceTransformerEmbeddings":
        embeder = SentenceTransformerEmbeddings(config["SentenceTransformerEmbeddings"], datafolder)
    elif model=="LlamaindexEmbeddings":
        embeder = LlamaIndexEmbeddings(config["LlamaindexEmbeddings"], datafolder)
    else:
        logger.error(f"Embeddings model {model} not suported")
    embeder.create_embeddings()

def do_retrieve(datafolder):
    logger.info(f"Working as retriever from context folder: {datafolder}")

def do_assistant(datafolder):
    logger.info(f"Working as Asistant from RAG context: {datafolder}")
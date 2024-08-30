import log

from LlamaIndexEmbeddings import LlamaIndexEmbeddings
from SentenceTransformerEmbeddings import SentenceTransformerEmbeddings
from LlamaIndexRetriever import LlamaIndexRetriever
from SentenceTransformerRetriever import SentenceTransformerRetriever
from RAGAssistant import RAGAssistant


def do_embeddings(datafolder):
    error=False
    model = log.config["global"]["embeddings"]
    log.logger.info(f"Working creating Embeddings (with {model}) from data folder: {datafolder}")
    if model=="SentenceTransformerEmbeddings":
        embeder = SentenceTransformerEmbeddings(log.config["SentenceTransformerEmbeddings"], datafolder)
    elif model=="LlamaIndexEmbeddings":
        embeder = LlamaIndexEmbeddings(log.config["LlamaindexEmbeddings"], datafolder)
    else:
        log.logger.error(f"Embeddings model {model} not suported")
        error=True
    if not error:
        embeder.create_embeddings()

def do_retrieve(datafolder, prompt_filename='', output_filename=''):
    error=False
    model = log.config["global"]["retriever"]
    log.logger.info(f"Working as retriever (with {model}) from data folder: {datafolder}")
    if model=="SentenceTransformerRetriever":
        retriever = SentenceTransformerRetriever(log.config["SentenceTransformerRetriever"], datafolder)
        retriever.load_embeddings()
    elif model=="LlamaIndexRetriever":
        retriever = LlamaIndexRetriever(log.config["LlamaIndexRetriever"], datafolder)
        retriever.load_embeddings()
    else:
        log.logger.error(f"Retriever model {model} not suported")
        error=True

    if not error:
        if output_filename=='':
            output_filename="output.txt"

        if prompt_filename!='':
            with open(prompt_filename) as input_f:
                with open(output_filename, "w") as output_f:
                    for line in input_f:
                        result = retriever.retrieve(line, True)
                        output_f.write(result)
        else: 
            prompt=input("Prompt: ")
            while (prompt!='exit'):
                result = retriever.retrieve(prompt, True)
                print(result)
                prompt = input("Prompt: ")

def do_assistant(datafolder):
    error=False
    model = log.config["global"]["assistant"]

    if model=="RAGAssistant":
        context=log.config["RAGAssistant"]["RAG_DB_Folder"]
        log.logger.info(f"Working as Asistant (with {model}) from RAG context: {context}")
        assistant = RAGAssistant(log.config["RAGAssistant"])
    #elif model=="TutoBotAssistant":
        #context=config["TutoBotAssistant"]["RAG_DB_Folder"]
        #logger.info(f"Working as Asistant (with {model}) from RAG context: {context}")
        #assistant = TutoBotAssistant(config["TutoBotAssistant"], datafolder)
    else:
        log.logger.error(f"Assistant {model} not suported")
        error=True
    
    if not error: #TODO: bucle fins 'exit', guardar històric   
        history=[]
        prompt=input("Prompt: ")
        while (prompt!='exit'):
            result = assistant.prompt(prompt,history)
            #TODO: history.add({'prompt':prompt, 'response':result})
            print(result)
            prompt = input("Prompt: ")
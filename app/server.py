import sys, getopt
from log import config, logger

from RAGAssistant import RAGAssistant
from ChatServer import ChatServer

def help():
    print ('server.py -d folder')

def do_server(datafolder):
    error=False
    model = config["global"]["assistant"]

    if model=="RAGAssistant":
        context=config["RAGAssistant"]["RAG_DB_Folder"]
        logger.info(f"Working as Asistant (with {model}) from RAG context: {context}")
        assistant = RAGAssistant(config["RAGAssistant"])
    #elif model=="TutoBotAssistant":
        #context=config["TutoBotAssistant"]["RAG_DB_Folder"]
        #logger.info(f"Working as Asistant (with {model}) from RAG context: {context}")
        #assistant = TutoBotAssistant(config["TutoBotAssistant"], datafolder)
    else:
        logger.error(f"Assistant {model} not suported")
        error=True
    if (not error):
        chat = ChatServer(assistant)
        chat.run()

def main(argv):
    # Argumentos
    # Modo de funcionamiento: -m [embeddings|retriever|assistant]
    # Datos: -d folder (servicios, cau, tutobot)

    opts, aux = getopt.getopt(argv,"hd:")
    error=False
    msg_error=""
    
    dataFolder = "serviciosUA"

    for (opt, arg) in opts:

        if opt == '-h':
            help()
            sys.exit()
        elif opt == "-d":
            if arg != None:
                dataFolder = arg
            else:
                msg_error="Error, you must enter a data folfer with argument '-d'."
                error=True
    
    if (error):
        print(msg_error)
        help()
    else:
        do_server(dataFolder)

if __name__ == "__main__":
   main(sys.argv[1:])
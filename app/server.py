import sys, getopt
import log

from RAGAssistant import RAGAssistant
from ChatServer import ChatServer

def help():
    print ('server.py -d folder')

def do_server(datafolder):
    error=False
    model = log.config["global"]["assistant"]

    if model=="RAGAssistant":
        context=log.config["RAGAssistant"]["RAG_DB_Folder"]
        log.logger.info(f"Working as Asistant (with {model}) from RAG context: {context}")
        assistant = RAGAssistant(log.config["RAGAssistant"])
    #elif model=="TutoBotAssistant":
        #context=log.config["TutoBotAssistant"]["RAG_DB_Folder"]
        #log.logger.info(f"Working as Asistant (with {model}) from RAG context: {context}")
        #assistant = TutoBotAssistant(log.config["TutoBotAssistant"], datafolder)
    else:
        log.logger.error(f"Assistant {model} not suported")
        error=True
    if (not error):
        chat = ChatServer(assistant)
        chat.run()

def main(argv):
    # Argumentos
    # Modo de funcionamiento: -m [embeddings|retriever|assistant]
    # Datos: -d folder (servicios, cau, tutobot)

    opts, aux = getopt.getopt(argv,"hd:c:")
    error=False
    msg_error=""
    
    dataFolder = "serviciosUA"
    configFile=""

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
        elif opt == "-c":
            if arg != None:
                configFile = arg
            else:
                msg_error="Error, you must enter a config file with argument '-c'."
                error=True
    
    if (error):
        print(msg_error)
        help()
    else:
        log.logger = log.Log(configFile)
        log.config = log.logger.getConfig()
        do_server(dataFolder)

if __name__ == "__main__":
   main(sys.argv[1:])
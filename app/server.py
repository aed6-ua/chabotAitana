import sys, getopt
from log import config, logger

def help():
    print ('server.py -m [embeddings|retriever|assistant] -d folder')

def do_work(mode, datafolder):
    if mode=="embeddings":
        logger.info(f"Working creating Embeddings from data folder: {datafolder}")
    elif mode=="retriever":
        logger.info(f"Working as retriever from context folder: {datafolder}")
    elif mode=="assistant":
        logger.info(f"Working as Asistant from RAG context: {datafolder}")
    else:
        msg_error="Error, value not valid for '-m'. Valid values are: embeddings, retriever, assistant"
        print(msg_error)
        help()

def main(argv):
    # Argumentos
    # Modo de funcionamiento: -m [embeddings|retriever|assistant]
    # Datos: -d folder (servicios, cau, tutobot)

    opts, args = getopt.getopt(argv,"hm:d:")
    error=False
    msg_error=""
    
    mode = "assistant"
    dataFolder = "serviciosUA"
    for opt, arg in opts:
        if opt == '-h':
            help()
            sys.exit()
        elif opt == "-m":
            if arg != None:
                mode = arg
                if (mode not in ('embeddings', 'retriever', 'assistant')):
                    msg_error="Error, value not valid for '-m'. Valid values are: embeddings, retriever, assistant"
                    error=True
            else:
                msg_error="Error, you must enter a working mode with argument '-m'."
                error=True
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
        do_work(mode, dataFolder)

if __name__ == "__main__":
   main(sys.argv[1:])
import sys, getopt
from log import config, logger
from util import do_embeddings, do_retrieve, do_assistant

def help():
    print ('server.py -m [embeddings|retriever|assistant] -d folder')

def do_work(mode, datafolder):
    if mode=="embeddings":
        do_embeddings(datafolder)
    elif mode=="retriever":
        do_retrieve(datafolder)
    elif mode=="assistant":
        do_assistant(datafolder)
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
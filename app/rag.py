import sys, getopt
from log import config, logger
from util import do_embeddings, do_retrieve, do_assistant

def help():
    print ('rag.py -m [embeddings|retriever|assistant] -d folder [-p prompt_filename]')

def do_work(mode, datafolder, prompt_filename=''):
    if mode=="embeddings":
        do_embeddings(datafolder)
    elif mode=="retriever":
        do_retrieve(datafolder, prompt_filename)
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

    opts, aux = getopt.getopt(argv,"hm:d:")
    error=False
    msg_error=""
    prompt_filename=''
    
    mode = "assistant"
    dataFolder = "serviciosUA"

    for (opt, arg) in opts:

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
        elif opt == "-p":
            if arg!= None:
                prompt_filename = arg
            else: 
                msg_error="Error, you must enter a prompt file name with argument '-p'."
    
    if (error):
        print(msg_error)
        help()
    else:
        do_work(mode, dataFolder, prompt_filename)

if __name__ == "__main__":
   main(sys.argv[1:])
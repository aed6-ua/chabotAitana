import glob
import pickle
from pathlib import Path

from log import config, logger

from sentence_transformers import SentenceTransformer, util

class SentenceTransformerEmbeddings():
    def __init__(self, local_config, datafolder):
            self.index_path = local_config["vector_folder"] + datafolder
            self.data_path = local_config["data_folder"] + datafolder
            self.model_name = local_config["model_name"]
            self.model_folder = local_config["model_folder"]
            self.chunk_size = local_config["chunk_size"]
            self.chunk_overlap = local_config["chunk_overlap"]
            self.data_filename = local_config["data_filename"]

            logger.info("Initializing SentenceTransformerEmbbedings...")
            self.model = SentenceTransformer(self.model_name, cache_folder=self.model_folder)
            logger.info(f"Loaded SentenceTransformer model for embedding: {self.model_name}")

            self.corpus_embeddings = []
            self.corpus_texts = []

    def create_embeddings(self): #TODO: fer!
        # load documents
        self.load_documents()
        # Crear embeddings y añadirlos a corpus_embeddings
        self.encode_embeddings()
        # guardar la BD
        self.save_embeddings()

    def load_documents(self):
        for filepath in glob.glob(self.data_path + "/*"):
            with open(filepath, 'r', encoding=config["global"]["data_encoding"]) as f:
                # Leemos todo el contenido del archivo en una sola cadena
                content = f.read()

                # Dividimos el contenido en párrafos
                paragraphs = content.split('\n\n')

                # Extendemos nuestra lista global de fragmentos de texto con estas oraciones
                self.corpus_texts.extend(paragraphs)
                
                logger.info(f"File {filepath} chunked.")

    def encode_embeddings(self):
        self.corpus_embeddings = self.model.encode(self.corpus_texts, convert_to_tensor=True)
        logger.info(f"Embeddings encoded.")

    def save_embeddings(self):
        data_to_save = {
            'corpus_embeddings': self.corpus_embeddings,
            'corpus_texts': self.corpus_texts
        }
        file_path = Path(self.index_path + "/" + self.data_filename)

        if not file_path.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.touch()

        with open(self.index_path + "/" + self.data_filename, "wb") as f:
            # Uso de pickle para guardar los datos en el disco de forma serializada.
            pickle.dump(data_to_save, f)
        logger.info(f"Embeddings saved to " + self.index_path + "/" + self.data_filename)


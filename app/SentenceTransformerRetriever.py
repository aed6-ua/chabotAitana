from sentence_transformers import SentenceTransformer, util

import pickle

#from log import config, logger
import log

class SentenceTransformerRetriever():
    def __init__(self, local_config, datafolder):
        self.index_path = local_config["vector_folder"] + datafolder + "/" + local_config["data_filename"]
        self.model_name = local_config["model_name"]
        self.model_folder = local_config["model_folder"]
        self.top_k = local_config["num_chunks"]
        self.min_relevance = local_config["%_sim_relevance"]

    def load_embeddings(self):
        self.model = SentenceTransformer(model_name_or_path=self.model_name, cache_folder=self.model_folder)
        log.logger.info(f"Loaded SentenceTransformer model: {self.model_name}")
        
        # Load the embeddings from the file
        with open(self.index_path, 'rb') as file:
            loaded_data = pickle.load(file)
        
        # Restaura los embeddings y textos desde el diccionario
        self.corpus_embeddings = loaded_data['corpus_embeddings']
        log.logger.info(f"Loaded {len(self.corpus_embeddings)} embeddings.")
        
        self.corpus_texts = loaded_data.get('corpus_texts', [])
        log.logger.info(f"Loaded {len(self.corpus_texts)} texts.")

    def retrieve(self, query, debug=False): #TODO: comparar amb el meu codi, eliminar chunks baix l'umbral
        
        if (self.corpus_embeddings is None):
            self.load_embeddings()
        
        
        # Tokenize the query
        query_embedding = self.model.encode(query, convert_to_tensor=True, show_progress_bar=False)
        
        # Compute the cosine similarity between the query and the corpus
        hits = util.semantic_search(query_embedding, self.corpus_embeddings, top_k=self.top_k)[0]
        # Format the results as a list of tuples
        results = [(self.corpus_texts[hit['corpus_id']], hit['score']) for hit in hits] #if self.corpus_texts_es else [(self.corpus_texts_en[hit['corpus_id']], hit['score']) for hit in hits]
        
        result=""
        if debug:
            num = 0
            result=""
            for node in results:
                result +=f"Text({(num+1)}) = " + node[0] + f"\nScore({(num+1)})={node[1]}\n"
                num=num+1
            result = f"Number of chunks: {num}.\n" + result
        else:
            result = '"""'
            for node in results:
                result += node[0] + "\n\n"
            result = result + '"""\n\n'
        return result


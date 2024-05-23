from abc import ABC, abstractmethod
from llama_index.core import StorageContext, load_index_from_storage
from langchain_community.embeddings import HuggingFaceEmbeddings
from llama_index.core.postprocessor import SimilarityPostprocessor


from log import config, logger

class LlamaIndexRetriever():
    def __init__(self, local_config, datafolder):
        self.index_path = local_config["vector_folder"] + datafolder
        self.model_name = local_config["model_name"]
        self.model_folder = local_config["model_folder"]
        self.top_k = local_config["num_chunks"]
        self.min_relevance = local_config["%_sim_relevance"]

    def load_embeddings(self):
        self._model = HuggingFaceEmbeddings(model_name=self.model_name, cache_folder=self.model_folder)
        self.storage_context = StorageContext.from_defaults(persist_dir=self.index_path)
        #Load index
        self.index = load_index_from_storage(self.storage_context, embed_model=self._model)
        logger.info("LlamaIndex embeddings loaded successfully.")
    
    def retrieve(self, query, debug=False):
        if (self.index is None):
            self.load_embeddings()
        retriever = self.index.as_retriever(similarity_top_k=self.top_k)
        # Use llamaindex to find the most relevant documents for the query
        raw_results = retriever.retrieve(query)

        processor = SimilarityPostprocessor(similarity_cutoff=(self.min_relevance/100))
        results = processor.postprocess_nodes(raw_results)

        num = 0
        result=""
        if debug:
            for node in results:
                result +=f"Text({(num+1)}) = " + node.get_text() + f"\nScore({(num+1)})={node.get_score()}\n"
                num=num+1
            result = f"Number of chunks: {num}.\n" + result
        else:
            result = '"""'
            for node in results:
                result += node[0] + "\n\n"
            result = result + '"""\n\n'
        
        return result
        

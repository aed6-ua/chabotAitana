from abc import ABC, abstractmethod
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the Retrieval Strategy Interface
class RetrievalStrategy(ABC):
    @abstractmethod
    def retrieve(self, query):
        pass

#TODO: vore si eliminar esta clase...
# class Retriever:
#     def __init__(self, strategy: RetrievalStrategy):
#         self.strategy = strategy

#     def load_embeddings(self):
#         # Placeholder for loading the corpus
#         pass

#     def retrieve(self, query):
#         """
#         Retrieves information based on the given query.
        
#         :param query: The query for which to retrieve information.
#         :return: Retrieved information or results.
#         """
#         # Placeholder for retrieval logic, e.g., database lookup, web search, etc.
#         return f"Information related to {query}"

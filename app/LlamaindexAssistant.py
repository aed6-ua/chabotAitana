import logging
from Assistant import Assistant
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.prompts.base import ChatPromptTemplate

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LlamaindexAssistant(Assistant):
    def __init__(self, model_name="gpt-3.5-turbo", retrieval_tool=None, api_parameters=None):
        super().__init__(tools=[retrieval_tool] if retrieval_tool is not None else [])
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        # load index
        self.index = load_index_from_storage(storage_context)
        # create query engine
        self.query_engine = self.index.as_query_engine()
        # Custom propts
        SYSTEM_PROMPT = ChatMessage(
            content=(
                "You are Aitana, an expert Q&A system from University of Alicante that is trusted around the world.\n"
                "Always answer the query using the provided context information, "
                "and not prior knowledge.\n"
                "Some rules to follow:\n"
                "1. Never directly reference the given context in your answer.\n"
                "2. Avoid statements like 'Based on the context, ...' or "
                "'The context information ...' or anything along "
                "those lines."
            ),
            role=MessageRole.SYSTEM,
        )
        PROMPT_TMPL_MSGS = [
            SYSTEM_PROMPT,
            ChatMessage(
                content=(
                    "Context information is below.\n"
                    "---------------------\n"
                    "{context_str}\n"
                    "---------------------\n"
                    "Given the context information and not prior knowledge, "
                    "answer the query in the same language.\n"
                    "Query: {query_str}\n"
                    "Answer: "
                ),
                role=MessageRole.USER,
            ),
        ]
        CHAT_PROMPT = ChatPromptTemplate(message_templates=PROMPT_TMPL_MSGS)

        self.query_engine.update_prompts({"response_synthesizer:text_qa_template":CHAT_PROMPT})
        prompts_dict = self.query_engine.get_prompts()
        for key in prompts_dict:
            print(key)
            print(prompts_dict[key])

    def process_message(self, message, context):
        try:
            response = self._generate_response(message)
            return response
        except Exception as e:
            logging.error(f"Error processing message: {e}")
            return "I'm sorry, I encountered an error processing your request."

    def _generate_response(self, prompt):
        response = self.query_engine.query(prompt).response
        return response
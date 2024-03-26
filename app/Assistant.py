from abc import ABC, abstractmethod
from log import config, logger

from openai import OpenAI

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.prompts.base import ChatPromptTemplate

#############################################################################################
#############################################################################################
#############################################################################################
class Assistant(ABC):
    def __init__(self, tools=None, prompt_settings=None):
        self.tools = tools if tools is not None else []

    @abstractmethod
    def process_message(self, message, context):
        """
        Processes a received message using the provided context.
        
        :param message: The message to process.
        :param context: The context of the conversation.
        :return: The response message.
        """
        pass

#############################################################################################
#############################################################################################
#############################################################################################
class GPTAssistant(Assistant):
    def __init__(self, model_name="gpt-3.5-turbo", retrieval_tool=None, prompt_settings=None, api_parameters=None):
        super().__init__(tools=[retrieval_tool] if retrieval_tool is not None else [])
        self.client = OpenAI()
        self.model_name = model_name
        self.prompt_settings = prompt_settings if prompt_settings is not None else {
            "language": "es",
            "introduction": "Eres Aitana, el asistente virtual de la Universidad de Alicante. Estas aquí para ayudar con información acerca de la universidad, incluyendo detalles sobre admisiones, programas académicos, eventos en el campus, servicios estudiantiles y más. Tu objetivo es proporcionar respuestas precisas y útiles a tus preguntas. Utiliza los textos proporcionados delimitados por comillas triples para responder preguntas. Si no se puede encontrar la respuesta en los textos, escribe 'No pude encontrar una respuesta'.",
            "context": ""
        }
        self.api_parameters = api_parameters if api_parameters is not None else {
            "temperature": 0.7,
            "max_tokens": 150
        }

    def process_message(self, message, context):
        try:
            enhanced_context = self._use_retrieval_tool_if_available(message, context)
            response = self._generate_response(message, enhanced_context)
            return response
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return "I'm sorry, I encountered an error processing your request."

    def _use_retrieval_tool_if_available(self, message, context):
        if self.tools and self.tools[0]:
            try:
                # List of tuples with the retrieved passages and their scores
                retrieval_result = self.tools[0].retrieve(message)
                # Convert to string with each result between triple quotes
                return "\n\n".join([f'"""{result}"""' for result, _ in retrieval_result])
            except Exception as e:
                logger.error(f"Retrieval tool failed: {e}")
        return context

    def _generate_response(self, message, context):
        message = f"CONTEXTO:\n{context}\n\nPREGUNTA:\n{message}"
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.prompt_settings["introduction"]},
                    {"role": "user", "content": message},
                ],
                model=self.model_name,
            )
            response = chat_completion.choices[0].message.content
            return response.strip()
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            # Instead of just raising the exception, we handle it gracefully
            return "I'm sorry, I encountered an error trying to generate a response. Please try again later."
        
#############################################################################################
#############################################################################################
#############################################################################################
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
            logger.error(f"Error processing message: {e}")
            return "I'm sorry, I encountered an error processing your request."

    def _generate_response(self, prompt):
        response = self.query_engine.query(prompt).response
        return response


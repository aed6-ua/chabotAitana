from Assistant import Assistant
from openai import OpenAI
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
            logging.error(f"Error processing message: {e}")
            return "I'm sorry, I encountered an error processing your request."

    def _use_retrieval_tool_if_available(self, message, context):
        if self.tools and self.tools[0]:
            try:
                # List of tuples with the retrieved passages and their scores
                retrieval_result = self.tools[0].retrieve(message)
                # Convert to string with each result between triple quotes
                return "\n\n".join([f'"""{result}"""' for result, _ in retrieval_result])
            except Exception as e:
                logging.warning(f"Retrieval tool failed: {e}")
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
            logging.error(f"Failed to generate response: {e}")
            # Instead of just raising the exception, we handle it gracefully
            return "I'm sorry, I encountered an error trying to generate a response. Please try again later."

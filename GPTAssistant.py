from Assistant import Assistant
from Model import OpenAIGenerationModel
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GPTAssistant(Assistant):
    def __init__(self, model=None, retrieval_tool=None, prompt_settings=None):
        super().__init__(tools=[retrieval_tool] if retrieval_tool is not None else [])
        self.prompt_settings = prompt_settings if prompt_settings is not None else {
            "language": "es",
            "introduction": "Eres Aitana, el asistente virtual de la Universidad de Alicante. Estas aquí para ayudar con información acerca de la universidad, incluyendo detalles sobre admisiones, programas académicos, eventos en el campus, servicios estudiantiles y más. Tu objetivo es proporcionar respuestas precisas y útiles a tus preguntas. Utiliza los textos proporcionados delimitados por comillas triples para responder preguntas. Si no se puede encontrar la respuesta en los textos, escribe 'No pude encontrar una respuesta'.",
            "context": ""
        }
        if model is None:
            self.model = OpenAIGenerationModel()
        else:
            self.model = model
        

    def process_message(self, message, context):
        try:
            enhanced_context = self._use_retrieval_tool_if_available(message, context)
            message = f"CONTEXTO:\n{enhanced_context}\n\nPREGUNTA:\n{message}"
            messages=[
                    {"role": "system", "content": self.prompt_settings["introduction"]},
                    {"role": "user", "content": message},
                ]
            try:
                return self.model.run(messages)
            except Exception as e:
                logging.error(f"Failed to generate response: {e}")
                # Instead of just raising the exception, we handle it gracefully
                return "I'm sorry, I encountered an error trying to generate a response. Please try again later."
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

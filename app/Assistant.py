from abc import ABC, abstractmethod

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

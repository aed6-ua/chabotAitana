class Conversation:
    def __init__(self, id, window_size=5):
        self.id = id
        self.messages = []
        self.window_size = window_size

    def add_message(self, sender, message):
        self.messages.append({"sender": sender, "message": message})

    def get_context(self):
        return self.messages[-self.window_size:]

    def __str__(self):
        # Return the conversation as a string
        return str(self.messages)
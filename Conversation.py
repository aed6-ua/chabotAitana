class Conversation:
    def __init__(self, id, window_size=5):
        self.id = id
        self.window_size = window_size
        self.star_rating = -1
        self.messages = []
        

    def add_message(self, sender, message):
        self.messages.append({"sender": sender, "message": message})

    def get_context(self):
        return self.messages[-self.window_size:]
    
    def stars(self, stars):
        self.star_rating = stars

    def __str__(self):
        self.messages.insert(0, {"sender": "rating", "value": self.star_rating})
        # Return the conversation as a string
        return str(self.messages)
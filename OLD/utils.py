from llama_index import VectorStoreIndex, SimpleDirectoryReader


def create_simple_llama_index():
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist()

class SimpleTextSplitter:
    def __init__(self, max_length=512, separators=[".", "!", "?"], overlap=0.25):
        self.max_length = max_length
        self.separators = separators
        self.overlap = overlap

    def split(self, text: str):
        # Split the text into maximum length segments. The segments must end with a separator. If a segment is too long, it is split at the last separator before the maximum length.
        # The overlap parameter determines the amount of overlap between segments. For example, if overlap is 0.25, the segments will overlap by 25%.
        # If no separator is found within the maximum length, the segment is split at the maximum length.
        segments = []
        start = 0
        end = 0
        while end < len(text):
            end = min(start + self.max_length, len(text))
            if end < len(text):
                # Find the last occurrence of a separator within the maximum length
                for i in range(end, start, -1):
                    if text[i-1] in self.separators:
                        end = i
                        break
            segments.append(text[start:end])
            start = int(end - self.overlap * self.max_length)
        return segments
    
    
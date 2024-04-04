import chromadb
client = chromadb.HttpClient(host='localhost', port=8000)
collection = client.get_collection(name="614477c6-cc76-4305-a4d7-567769d8833f")
print(collection.peek())
import chromadb
client = chromadb.PersistentClient(path="/tmp/chroma_db_dsw")
collection = client.get_collection("rag_knowledge_base")
result = collection.get(limit=1, include=["metadatas"])
print(result["metadatas"])
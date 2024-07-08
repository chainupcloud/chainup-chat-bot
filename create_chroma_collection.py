import chromadb
from chromadb.utils import embedding_functions
from load_data import load_pdf_data

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

print("Creating ChromaDB client...")
chroma_client = chromadb.PersistentClient(path="db/")

print("Getting or creating collection...")
collection = chroma_client.get_or_create_collection(name="chainup_intro", embedding_function=sentence_transformer_ef)

print("Loading PDF data...")
docs, ids = load_pdf_data('./training_pdfs')

print("Adding documents to collection...")
existing_ids = collection.get(ids=ids)["ids"]
new_docs = [doc for doc, id in zip(docs, ids) if id not in existing_ids]
new_ids = [id for id in ids if id not in existing_ids]

try:
    print("Adding new documents to collection...")
    collection.add(documents=new_docs, ids=new_ids)
    print("Success! New documents added to collection.")
except Exception as e:
    print(f"One or more of the documents you tried to add already exist in the collection. Skipping duplicate documents: {e}")

# Query the collection
# query_dict = {"$contains": "What is the main topic of this document?"}
# results = collection.get(
#     where_document=query_dict,
#     limit=3,
#     include=["documents", "metadatas", "embeddings"]
# )

# # Print the results
# for i, result in enumerate(results["documents"]):
#     print(result)
#     print("Metadata:", results["metadatas"][i])
#     print("Embedding:", results["embeddings"][i])
#     print()

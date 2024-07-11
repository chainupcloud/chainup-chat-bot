import chromadb
from chromadb.utils import embedding_functions
from langchain_community.document_loaders import PyPDFLoader
from load_data import load_json_data

print(" instantiate the Chroma DB vector store.")
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="db/")
collection = chroma_client.get_or_create_collection(name="chainup", embedding_function=sentence_transformer_ef)

print(" # load the document, split it into pages, and then prepare it for insertion in the vector store.")
pdf_loader = PyPDFLoader('./training_pdfs/ChainUp Introduction.pdf')
documents = pdf_loader.load_and_split()

# print("# call the ChromaDB function to add documents to the vector store and generate embeddings on the fly ")
collection.add(
    documents=[doc.page_content for doc in documents],
    ids=[str(i) for i in range(len(documents))]
)

# Load the JSON data
json_dir = './training_jsons'
print(f"Loading JSON data from directory: {json_dir}")
docs, ids = load_json_data(json_dir)

# Print information about loaded data (optional)
print(f"Number of json data loaded: {len(docs)}")
# Add the JSON data to the collection
collection.add(documents=docs, ids=ids)
import chromadb
from chromadb.utils import embedding_functions
from langchain_community.document_loaders import PyPDFLoader

print(" instantiate the Chroma DB vector store.")
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="db/")
collection = chroma_client.get_or_create_collection(name="city_laws", embedding_function=sentence_transformer_ef)

print(" # load the document, split it into pages, and then prepare it for insertion in the vector store.")
pdf_loader = PyPDFLoader('./training_pdfs/ChainUp Introduction.pdf')
documents = pdf_loader.load_and_split()
print(len(documents))

print(" # This uses the LangChain utility function to load and split the document. ")
docs = []
ids = []
i = 0
for doc in documents:
  docs.append(doc.page_content)
  ids.append(str(i))
  i = i+1

print("# call the ChromaDB function to add documents to the vector store and generate embeddings on the fly ")
collection.add(
    documents = docs,
    ids = ids
)
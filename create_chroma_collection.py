import chromadb
from chromadb.utils import embedding_functions
from load_data import load_json_ids
from load_data import load_uri_ids
from load_data import load_pdf_ids

print(" instantiate the Chroma DB vector store.")
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="db/")
collection = chroma_client.get_or_create_collection(name="Chainup", embedding_function=sentence_transformer_ef)


# Load the JSON data
json_dir = './training_jsons'
# json_dir = './training_jsons/insert'

print(f"Loading JSON data from directory: {json_dir}")
json_docs, json_ids = load_json_ids(json_dir)
print(f"Number of json data loaded: {len(json_docs)}")
# Add the JSON data to the collection
collection.add(documents=json_docs, ids=json_ids)


# #Load Webpage data
urls = [
# # #Load Webpage data
# urls = [
# "https://www.chainup.com/", "https://www.chainup.com/zh", "https://www.chainup.com/product/exchange", "https://www.chainup.com/zh/product/exchange", "https://www.chainup.com/product/dex", "https://www.chainup.com/zh/product/dex", "https://www.chainup.com/product/liquidity", "https://www.chainup.com/zh/product/liquidity", "https://www.chainup.com/product/wallet", "https://www.chainup.com/zh/product/wallet", "https://www.trustformer.info/zh-Hant", "https://www.trustformer.info", "https://www.chainup.com/solutions/central-banks", "https://www.chainup.com/zh/solutions/central-banks", "https://www.chainup.com/solutions/commercial-banks", "https://www.chainup.com/zh/solutions/commercial-banks", "https://www.chainup.com/about", "https://www.chainup.com/zh/about", "https://www.chainup.com/solution/AssetTokenization", "https://www.chainup.com/zh/solution/AssetTokenization", "https://www.chainup.com/solution/GamingandMetaverse", "https://www.chainup.com/zh/solution/GamingandMetaverse", "https://www.chainup.com/solution/SmartWeb3Banking", "https://www.chainup.com/zh/solution/SmartWeb3Banking", "https://custody.chainup.com/zh-HK/joint", "https://custody.chainup.com/joint", "https://custody.chainup.com/zh-HK/mpcPlatform", "https://custody.chainup.com/mpcPlatform", "https://www.chainup.com/product/investment", "https://www.chainup.com/zh/product/investment", "https://www.chainup.com/product/ipfs", "https://www.chainup.com/zh/product/ipfs"
# ]

# web_docs, web_ids = load_uri_ids(urls)
# # Add Webpage data to the collection
# collection.add(documents=[str(doc) for doc in web_docs], ids=web_ids)
# print(f"Document IDs: {web_ids}")


# # Load PDF data
# print(" # load the document, split it into pages, and then prepare it for insertion in the vector store.")
# pdf_docs, pdf_ids = load_pdf_ids('./training_pdfs')
# collection.add(documents=pdf_docs, ids=pdf_ids)
# print(f"PDF IDs: {pdf_ids}")


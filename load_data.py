import os
import chromadb
from chromadb.utils import embedding_functions
from langchain_community.document_loaders import PyPDFLoader
import json
import requests
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdf_ids(pdf_dir):
    docs = []
    ids = []
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, filename)
            pdf_loader = PyPDFLoader(pdf_path)
            documents = pdf_loader.load_and_split()
            for i, doc in enumerate(documents):
                docs.append(doc.page_content)
                ids.append(f"{filename}_{i}")
    return docs, ids


def load_json_ids(json_dir):
    docs = []
    ids = []
    for filename in os.listdir(json_dir):
        if filename.endswith(".json"):
            json_path = os.path.join(json_dir, filename)
            with open(json_path, 'r') as f:
                data = json.load(f)
                for i, obj in enumerate(data):
                    doc_text = ""
                    for key, value in obj.items():
                        doc_text += f"{key}: {value} "
                    docs.append(doc_text.strip())
                    ids.append(f"{filename}_{i}")
    return docs, ids

    import requests

def load_uri_ids(uris):
    ids = []
    docs_list = []
    for i, uri in enumerate(uris):
        response = requests.get(uri)
        if response.status_code == 200:
            ids.append(f"uri_{i}")
            web_doc = WebBaseLoader(uri).load()
            docs_list.extend(web_doc)
        else:
            print(f"Error loading URI {uri}: {response.status_code}")
    
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
    doc_splits = text_splitter.split_documents(docs_list)
    print(f"\nCollection stats: Number of documents: {len(doc_splits)}")
    
    return doc_splits, ids
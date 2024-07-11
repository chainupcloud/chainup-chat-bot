import os
import chromadb
from chromadb.utils import embedding_functions
from langchain_community.document_loaders import PyPDFLoader
import json

def load_pdf_data(pdf_dir):
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

def load_json_data(json_dir):
    docs = []
    ids = []
    for filename in os.listdir(json_dir):
        if filename.endswith(".json"):
            json_path = os.path.join(json_dir, filename)
            with open(json_path, 'r') as f:
                data = json.load(f)
                for i, obj in enumerate(data):
                    input_text = obj.get("input", "")
                    output_text = obj.get("output", "")
                    doc_text = f"{input_text} {output_text}"
                    docs.append(doc_text)
                    ids.append(f"{filename}_{i}")
    return docs, ids
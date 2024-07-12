# chainup-chat-bot
Privatized chatbots based on RAG and Llama3.

[简体中文](README-zh.md) | [English](README.md)

# Technical architecture
## Architecture diagram
![chatbot](diagrams/chatbot.png)

## Key technologies
1. RAG (Retrieval Enhanced Generation) refers to the optimization of large language model output so that it can reference an authoritative knowledge base outside of the training data source before generating a response. Large language models (LLMs) are trained on massive amounts of data, using billions of parameters to generate raw output for tasks such as answering questions, translating language, and completing sentences. Building on the already powerful capabilities of LLMs, RAG extends them to provide access to domain-specific or organization-specific internal knowledge bases, all without the need to retrain models. It's a cost-effective way to improve LLM output to keep it relevant, accurate, and useful in a variety of contexts.
    1. [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
    2. [What is RAG (Retrieval Augmented Generation)?](https://aws.amazon.com/cn/what-is/retrieval-augmented-generation/)
2. [Ollama](https://github.com/ollama/ollama) Get a framework to get large models up and running quickly。
3. [Llama3 8B](https://llama.meta.com/llama3/) Meta's open-source model.
4. [LangChain](https://www.langchain.com/) Help developers easily build applications based on large language models (LLMs).。

# Usage

## Env set

1. Install python dependencies
```sh
pip install dspy gradio langchain langchain_community langchain_core langchain_huggingface pypdf fastembed chromadb sentence-transformers pandas openpyxl
```

2. Ollama 

see：[https://github.com/ollama/ollama](https://github.com/ollama/ollama)
```sh
# 1. install Ollama: https://github.com/ollama/ollama

# 2. Run ollama
ollama serve

# 3. Download llama3 8B
ollama pull llama3

# 4. run
ollama run llama3
```

## Prepare data
Supported data types：
1. json（training_jsons）
2. pdf（training_pdfs）
3. xlsx（training_xlsx）
4. tweets

## The data is loaded into the vector database ChromaDB
```python
# Convert xlsx file to json
python training_xlsx_to_json.py

# Create a local ChromaDB (/db in the root directory of the project)
python create_chroma_collection.py

# Load training data to ChromaDB (new files can be added at any time)
python load_data.py
```

## Run Chatbot
```python
python chatbot.py
```

After running, open the browser to access: http://localhost:7860, which comes with a web UI interface.


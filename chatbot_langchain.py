import json
import gradio as gr

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Create the prompte from the template.
promptTemplate = """Answer the question as precise as possible using the provided context. If the answer is
    not contained in the context, say "answer not available in context" \n\n
    Context: {context}
    Question: {question}
    Answer:

     """
modelSel = ""


# Load the PDF file to ChromaDB
def load_data_from_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    chunks = filter_complex_metadata(pages)
    vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())
    # vector_store = Chroma.from_documents(documents=chunks, embedding=HuggingFaceEmbeddings())
    return vector_store


def model_response(message, history):
    # llm = ChatOllama(model = conf["model"])
    llm = Ollama(model="llama3")

    prompt = PromptTemplate(template=promptTemplate, input_variables=["context", "question"])

    # Initiate the retriever
    db_loaded = load_data_from_pdf("training_pdfs/ChainUp Introduction.pdf")
    retriever = db_loaded.as_retriever(search_type="similarity_score_threshold", search_kwargs={
        "k": 5,
        "score_threshold": 0.2
    })
    hp_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    return hp_chain.invoke(message)


if __name__ == "__main__":
    # read configuration file
    # conf = {}
    # with open("config.json" , "r") as confFile:
    #     conf = json.load(confFile)
    #     print(conf["model"])

    chatUI = gr.ChatInterface(fn=model_response,
                              title="Your Friendly Web3 Chatbot",
                              description="Ask me about anything about Web3 and ChainUp."
                              )
    chatUI.launch()

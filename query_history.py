# query_history.py

import gradio as gr
from chromadb.utils import embedding_functions
import chromadb

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="userqueries/")
queries_collection = chroma_client.get_or_create_collection(name="UserQueries", embedding_function=sentence_transformer_ef)
result = queries_collection.peek()
queries = result['documents']
print(queries)

css = ".gradio-container {background: linear-gradient(to bottom, rgba(255,255,255,0.9), rgba(240,240,240,0.9)), url(https://i.postimg.cc/Dy6CgyF8/Chain-UP.png) repeat-x left center; background-size: auto, cover; padding: 20px; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 100vh; }"

query_history_iface = gr.Interface(
    fn=lambda: "\n".join(queries),
    inputs=[],
    outputs=[gr.Textbox(label="Most Recent Queries", lines=10)],
    title="Query History",
    description="View all user queries submitted to the chatbot.",
    css=css
)

if __name__ == "__main__":
    query_history_iface.launch(share=True)
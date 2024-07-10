import chromadb
from chromadb.utils import embedding_functions
import gradio as gr
import dspy
from dspy.retrieve.chromadb_rm import ChromadbRM

print("# set up the DSPy module.")
retriever_model = ChromadbRM(
'city_laws', 'db/', embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2"), k=5
)

print("# set up gpt-3.5-turbo model.")
turbo = dspy.OpenAI(model='gpt-3.5-turbo')

dspy.settings.configure(lm=turbo, rm=retriever_model)
# print(turbo("Write a short poem")) 

print("# set up RAG module.")
class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")
    
    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)

uncompiled_rag = RAG()

# Add a Gradio UI
def chatbot_interface(user_input, history):
    response = uncompiled_rag(user_input)
    return f"{response.answer}\n{[c for c in response.context]}"

iface = gr.Interface(
    fn=chatbot_interface,
    inputs=[gr.Textbox(label="Answer", lines=3)], 
    outputs=[gr.Textbox(label="Answer", lines=3)],
    title="Your Friendly Web3 Chatbot",
    description="Ask me about anything about Web3 and Chainup."
)

iface.launch(share=True)
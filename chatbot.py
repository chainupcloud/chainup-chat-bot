import chromadb
from chromadb.utils import embedding_functions
import gradio as gr
import dspy
from dspy.retrieve.chromadb_rm import ChromadbRM
import dsp
import time

# chatbot.py
print("# set up the DSPy module.")
retriever_model = ChromadbRM(
    'Chainup', 'db/',
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2"), k=5
)

# gpt-3.5-turbo
model_name = "llama3"
# model_name = "gpt-3.5-turbo"
print(f"# set up model: {model_name}.")

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="userqueries/")
queries_collection = chroma_client.get_or_create_collection(name="UserQueries", embedding_function=sentence_transformer_ef)

turbo = None
# Set the turbo variable based on the model name
if "gpt" in model_name:
    turbo = dspy.OpenAI(model=model_name)
elif "llama" in model_name:
    turbo = dspy.OllamaLocal(model=model_name, model_type='text', max_tokens=350,  temperature=0.1, top_p=0.8, frequency_penalty=1.17, top_k=40)
else:
    raise ValueError(f"Unknown model: {model_name}")

dspy.settings.configure(lm=turbo, rm=retriever_model)

print("# set up RAG module.")
class GenerateAnswer(dspy.Signature): 
    """Answer questions with short factoid answers."""
    context = dspy.InputField(desc="may contain relevant facts or answer keywords")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="a natural language response, an answer between 1 and 40 words ")

class RAG(dspy.Module): 
    def __init__(self, num_passages=6):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
    
    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)

# Add a Gradio UI

uncompiled_rag = RAG()

def chatbot_interface(user_input):
    current_time = str(int(time.time()))
    response = uncompiled_rag(user_input)
    queries_collection.add(documents=[user_input], ids=[current_time] )
    return f"{response.answer}"

css = ".gradio-container {background: linear-gradient(to bottom, rgba(255,255,255,0.9), rgba(240,240,240,0.9)), url(https://i.postimg.cc/Dy6CgyF8/Chain-UP.png) repeat-x left center; background-size: auto, cover; padding: 20px; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 100vh; }"

iface = gr.Interface(
    fn=chatbot_interface,
    inputs=[gr.Textbox(label="Ask", lines=3)],
    outputs=[gr.Textbox(label="Answer", lines=3)],
    title="Your Friendly Web3 Chatbot",
    css=css
)

iface.launch(server_name="0.0.0.0", server_port=7860, share=True)

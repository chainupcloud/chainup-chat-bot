import dspy
from dspy.retrieve.chromadb_rm import ChromadbRM
from load_data import load_json_data
from dspy.teleprompt import BootstrapFewShot
import gradio as gr
from chromadb.utils import embedding_functions
from functools import lru_cache


# lm = dspy.OllamaLocal(model='llama3')
lm = dspy.OllamaLocal(model='llama2')
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

dspy.settings.configure(lm=lm, rm=ChromadbRM(
    'chainup_intro', 'db/', embedding_function=sentence_transformer_ef, k=3
))

class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")    

    @lru_cache(maxsize=128)
    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)

@lru_cache(maxsize=128)
def compiled_rag(user_input):
    response = RAG()(user_input)
    answer = response.answer
    # Summarize the answer using LLaMA2, limited to 30 tokens (approx. 20-25 words)
    summary = lm.summarize(answer, max_tokens=20)
    # summary = lm.summarize(answer, ratio=0.2)
    return summary

def validate_context_and_answer(example, pred, trace=None):
    answer_EM = dspy.evaluate.answer_exact_match(example, pred)
    answer_PM = dspy.evaluate.answer_passage_match(example, pred)
    return answer_EM and answer_PM

print("Loading training data...")
trainset_file_path = './training_jsons/trainset1.jsonl'
trainset = load_json_data(trainset_file_path)

print("Compiling teleprompter...")
teleprompter = BootstrapFewShot(metric=validate_context_and_answer)
compiled_rag = teleprompter.compile(RAG(), trainset=trainset)

def chatbot_interface(user_input, history):
    response = compiled_rag(user_input)
    answer = response.answer
    return answer

iface = gr.Interface(
    fn=chatbot_interface,
    inputs=gr.Textbox(label="User Input"),
    outputs=gr.Textbox(label="Response"),
    title="Chainup Chatbot",
    description="Welcome to ask me about all things Web3 and ChainUp!"
)

print("Launching Gradio interface...")
iface.launch(share=True)
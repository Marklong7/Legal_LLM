import os
import json
import pickle
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

from langchain import hub
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from google.auth import default
from dotenv import load_dotenv

# load_dotenv("../.env") 
openai_api_key = os.getenv("OPENAI_API_KEY")
print(openai_api_key)
# Load Raptor results from a pickle file
with open('saved_raptor_result.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

raptor = loaded_dict

# Load chunked_texts from the file
with open('chunked_texts.json', 'r') as f:
    chunked_texts = json.load(f)

# Prepare texts for embedding
all_texts = chunked_texts.copy()

# Iterate through the Raptor results to extract summaries and add them to all_texts
for level in sorted(raptor.keys()):
    summaries = raptor[level][1]["summaries"].tolist()
    all_texts.extend(summaries)

# Initialize embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Directory to store Chroma vector store
chroma_persist_directory = "./chroma_vector_store"

# Check if the Chroma vector store already exists
if os.path.exists(chroma_persist_directory):
    # Load the vector store if it exists
    vectorstore = Chroma(persist_directory=chroma_persist_directory, embedding_function=embeddings)
    print("Loaded Chroma vector store from disk.")
else:
    # Create the vector store and save it
    vectorstore = Chroma.from_texts(
        texts=all_texts,
        embedding=embeddings,
        persist_directory=chroma_persist_directory
    )
    vectorstore.persist()  # Save the vector store to disk
    print("Created and saved Chroma vector store to disk.")

# Create a retriever from the Chroma vector store
retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

# Load and upgrade the RAG prompt
rag_prompt = hub.pull("rlm/rag-prompt")
rag_prompt.messages[0].prompt.template = """
You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
Then, in a new line to list the reference, briefly tells the user the resource of the context, usually their titles are enough.

If you don't know the answer, just say that you don't know.
\nQuestion: {question} \nContext: {context} \nAnswer:"
"""

# Set up the LLM model
llm_model = ChatOpenAI(model_name="gpt-4", openai_api_key=openai_api_key, temperature=0.05)

# RAG Chain - set up to handle questions using retriever, prompt, and LLM
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | llm_model
    | StrOutputParser()
)

# Function to handle questions through RAG + Raptor chain
def question_handler(question):
    print(f'Q: {question}')
    answer = rag_chain.invoke(question)
    print(f'A: {answer}')
    return answer

# Test the Raptor Script functionality with a sample question
if __name__ == "__main__":
    test_question = "Give me some information about Illinois healthcare law."
    question_handler(test_question)

import os
import chainlit as cl
import tiktoken
import openai
from dotenv import load_dotenv
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain import text_splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from operator import itemgetter
from langchain.schema.runnable import RunnablePassthrough

# GLOBAL SCOPE - ENTIRE APPLICATION HAS ACCESS TO VALUES SET IN THIS SCOPE #
# ---- ENV VARIABLES ---- # 
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client after loading the environment variables
openai.api_key = OPENAI_API_KEY

# -- RETRIEVAL -- #
"""
1. Load Documents from Text File
2. Split Documents into Chunks
3. Push files into our vectorstore
"""
### 1. CREATE TEXT LOADER AND LOAD DOCUMENTS
source_file = PyMuPDFLoader("./data/airbnb-10k.pdf")
loaded_file = source_file.load()

def tiktoken_len(text):
    tokens = tiktoken.encoding_for_model("gpt-4o").encode(text)
    return len(tokens)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 30,
    length_function = tiktoken_len,
)
chunks = text_splitter.split_documents(loaded_file)

#-----Embedding and Vector Store Setup-----#
# Load OpenAI Embeddings Model
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Creating Qdrant Vector Store
qdrant_vector_store = Qdrant.from_documents(
    chunks,
    embeddings,
    location=":memory:",
    collection_name="airbnbdocs",
)

# Create a Retriever
retriever = qdrant_vector_store.as_retriever()

#-----Prompt Template and Language Model Setup-----#
# Define the prompt template
template = """Answer the query from the user based only on the context provided. \n
If you cannot answer the question with the context, please respond with 'I don't believe I have the answer to that question, could you try asking it again in a different way?'.
Please also provide examples form the context material that illustrate how you came to your answer.
Context:
{context}
Question:
{question}
"""

prompt = ChatPromptTemplate.from_template(template)
chat_llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
retrieval_augmented_qa_chain = (
    {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": prompt | chat_llm, "context": itemgetter("context")}
)

#-----Chainlit Integration-----#
@cl.on_chat_start  
async def start_chat():
    settings = {
        "model": "gpt-4o",
        "temperature": 0,
        "max_tokens": 500,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }
    cl.user_session.set("settings", settings)

# Processes incoming messages from the user and sends a response through a series of steps:
# (1) Retrieves the user's settings
# (2) Invokes the RAG chain with the user's message
# (3) Extracts the content from the response and sends it back to the user

@cl.on_message 
async def handle_message(message: cl.Message):
    settings = cl.user_session.get("settings")

    response = retrieval_augmented_qa_chain.invoke({"question": message.content})


    # Extracting and sending just the content
    content = response["response"].content
    pretty_content = content.strip()  # Remove any leading/trailing whitespace

    await cl.Message(content=pretty_content).send()

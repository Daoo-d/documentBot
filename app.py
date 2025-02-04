import os
import streamlit as st
from constants import HUGGINGFACEHUB_API_TOKEN, Pinecone_API_KEY

# Set up HuggingFace API Token (use your own token)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

def readdoc(directory):
    # Load the PDF document
    pdf_loader = PyPDFLoader(directory)
    file = pdf_loader.load()
    return file

def chunk_data(doc, chunk_size=800, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(doc)
    return docs

def create_vectors(text, embeddings, index_name):
    pc = Pinecone(
        api_key=Pinecone_API_KEY
    )
    index = pc.Index(index_name)

    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    vector_store.add_documents(text)
    return vector_store

def get_vector(text, embeddings, index_name):
    pc = Pinecone(
        api_key=Pinecone_API_KEY
    )
    index = pc.Index(index_name)

    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    return vector_store

def retrieve_query(query, vectors, k):
    document = vectors.similarity_search(query, k=k)
    return document

# Set the file path and embeddings
path = r"E:\Users\Dawood Ahmed\Downloads\Asssighmnent.pdf"
embeddings = HuggingFaceEndpointEmbeddings()

# Read and chunk the document
doc = readdoc(path)
text = chunk_data(doc)

# Set the index name and retrieve vectors
index_name = "llmvecdb"
vectors = get_vector(text, embeddings, index_name)

# Query and retrieve relevant document chunks
query = "TF-IDF"
res = retrieve_query(query, vectors, 2)

# Set up the LLM
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    temperature=0.5,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    model_kwargs={"max_length": 250},
)

# Define the prompt template
prompt = PromptTemplate(
    input_variables=['question', 'context'],
    template = """
    You are an intelligent assistant. Given the following context, answer the question as accurately and concisely as possible. Also add the phrase accorsing to the document at the start of the answer.

    Context:
    {context}

    Question:
    {question}

    Answer:

    """   
)

# Prepare the context and invoke the chain
context = "\n\n".join([doc.page_content for doc in res])
formatted_input = prompt.format(question=query, context=context)
response = llm.invoke(formatted_input)
print(response)

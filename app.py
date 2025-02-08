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
from langchain_core.documents import Document

def readdoc(directory):
    # Load the PDF document
    pdf_loader = PyPDFLoader(directory)
    documents = pdf_loader.load()

    # Clean the documents
    cleaned_documents = []
    for doc in documents:
        cleaned_page_content = clean_text(doc.page_content)
        cleaned_doc = Document(
            metadata=doc.metadata,
            page_content=cleaned_page_content
        )
        cleaned_documents.append(cleaned_doc)
    return cleaned_documents    
def clean_text(text):
    # Implement your text cleaning logic here
    cleaned_text = text.replace('\n', ' ').replace('\u200b', '').strip()
    return cleaned_text

def chunk_data(doc, chunk_size=1000, chunk_overlap=100):
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

def retrieve_query(query, vectors, k=5):
    document = vectors.similarity_search(query, k=k)
    return document
def get_document_overview(doc):
    overview = {
        "total_pages": len(doc),
        "titles": [page.title for page in doc if hasattr(page, 'title')],
        "sections": [page.section for page in doc if hasattr(page, 'section')]
    }
    return overview
# Set the file path and embeddings
path = r"E:\Users\Dawood Ahmed\Downloads\JOB.pdf"
embeddings = HuggingFaceEndpointEmbeddings()

# Read and chunk the document
doc = readdoc(path)
text = chunk_data(doc)
# Set the index name and retrieve vectors
index_name = "llmvecdb"
vectors = get_vector(text, embeddings, index_name)

# Query and retrieve relevant document chunks
query = "what is the document for "
res = retrieve_query(query, vectors, 5)
print(res)

# Set up the LLM
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    temperature=0.3,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    model_kwargs={"max_length": 300},
)

# Define the prompt template
prompt = PromptTemplate(
    input_variables=['question', 'context'],
    template = """
    You are an intelligent assistant. Given the following context, answer the question as accurately and concisely as possible. Also add the phrase according to the document at the start of the answer.

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

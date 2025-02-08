import streamlit as st
from pypdf import PdfReader
import time
from pinecone import Pinecone
from langchain_community.vectorstores import FAISS
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from constants import HUGGINGFACEHUB_API_TOKEN, Pinecone_API_KEY
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain.schema import HumanMessage,AIMessage,SystemMessage

 

def create_chunks(doc):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len)
    chunks = splitter.split_text(doc)
    return chunks

def extract_text_from_pdf(pdf_files):
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()
    return text          

def create_vectors(chunks, index_name):
    embeddings = HuggingFaceEndpointEmbeddings()
    pc = Pinecone(
        api_key = Pinecone_API_KEY
    )
    index = pc.Index(index_name)
    vector_store = PineconeVectorStore(index=index,embedding=embeddings)
    documents = [Document(
            metadata={'id': str(i)},
            page_content=chunk
        ) for i, chunk in enumerate(chunks)]
    vector_store.add_documents(documents)
    return vector_store

def get_vector_response(text, index_name, k=5):
    embeddings = HuggingFaceEndpointEmbeddings()

    pc = Pinecone(
        api_key=Pinecone_API_KEY
    )
    index = pc.Index(index_name)
    vector_store = PineconeVectorStore(index=index,embedding=embeddings)
    response = vector_store.similarity_search(text, k)
    context = ''.join([doc.page_content for doc in response])
    return context

def is_document_related(query):
    # Simple heuristic: Check if the query contains certain keywords
    document_keywords = ["document", "content", "section", "chapter", "topic"]  # Add relevant keywords for your domain
    return any(keyword in query.lower() for keyword in document_keywords)


def get_chat_response(query):
    repo_id = "HuggingFaceH4/zephyr-7b-alpha"  # HuggingFace model (change this as needed)

    # Initialize the HuggingFace endpoint
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        task="text-generation",
        temperature=0.7,
        do_sample=False,
        repetition_penalty=1.03,
        max_new_tokens=512,
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    )
    chat = ChatHuggingFace(llm=llm)
    # Append the user query to the chat history (HumanMessage)
    st.session_state['chat_history'].append(HumanMessage(content=query))

    # Determine if the query is document-related or a general query
    if is_document_related(query):
        # Retrieve relevant context from the document store (Pinecone or FAISS)
        context = get_vector_response(query, index_name="embedvectors")
        print("this is the content of the sys:",context)
        # Define the prompt template for the model with chat history included
        prompt_template = """
    You are an intelligent assistant. Given the following context, answer the question as accurately and concisely as possible. Also add the phrase according to the document at the start of the answer.

    Previous conversation:
    {chat_history}

    Context:
    {context}

    Question:
    {question}
    
    
    Answer:

    """ 
        # Create the formatted prompt with context, chat history, and question
        prompt = prompt_template.format(
            context=context, 
            chat_history="\n".join([f"{msg.__class__.__name__}: {msg.content}" for msg in st.session_state['chat_history']]),
            question=query
        )

    else:
        # General question (not document-related)
        prompt_template = """
        You are a helpful assistant. Answer the following general question based on your knowledge:

        Previous conversation:
        {chat_history}

        Question:
        {question}

        Answer:
        """

        prompt = prompt_template.format(
            chat_history="\n".join([f"{msg.__class__.__name__}: {msg.content}" for msg in st.session_state['chat_history']]),
            question=query
        )

    # Send the prompt to the model and get the response
    response = chat.invoke(prompt)

    if isinstance(response, AIMessage):
        answer = response.content
    else:
        answer = response

    # Append the model's response (AIMessage) to the chat history
    st.session_state['chat_history'].append(AIMessage(content=answer))

    return answer



def main():
    st.set_page_config(page_title="ChatDoc", page_icon="🔍")
    st.header("Chat with your documents")
    q = st.text_input("Ask anything you want")
    rawtext = ""
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = [
        SystemMessage(content="""
You are a friendly and helpful chatbot, here to assist with both document-related and general questions. 
You will always have access to the previous conversation history, so respond in a conversational and engaging manner.

For document-related questions, you will receive context from the document. 
Answer the question in a clear and concise way based on the provided context. 
If you can't find the answer in the document, simply say "I can't find the answer in the document."

For general questions, use your own knowledge to provide a helpful response. 
If you're unsure about something, feel free to say "I don't know."

Just remember, you're here to have a helpful and friendly conversation, whether it's about the document or something else!
""")

    ]
    with st.sidebar:
        st.subheader("Upload document")
        filep = st.file_uploader("Upload your document", accept_multiple_files=True)
        submit = st.button("Process")
        
        if submit and filep:
            with st.spinner("Processing..."):
                start_time = time.time()
                # Extract text from PDF
                rawtext = extract_text_from_pdf(filep)

                # Create chunks
                chunks = create_chunks(rawtext)

                # Create vectorestores
                create_vectors(chunks, index_name="embedvectors")

                end_time = time.time()
                st.write(f"Processed in {end_time - start_time:.2f} seconds")
    if q:
        res = get_chat_response(q)
        st.write(res)


if __name__ == "__main__":
    main()
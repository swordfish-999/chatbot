import streamlit as st
import os
import sqlite3
import tempfile
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings


# API
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")



# Database Setup
conn = sqlite3.connect("chat_history.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT,
    message TEXT
)
""")
conn.commit()


def save_message(role, message):
    cursor.execute(
        "INSERT INTO chat_history (role, message) VALUES (?, ?)",
        (role, message)
    )
    conn.commit()


def load_chat_history():
    cursor.execute("SELECT role, message FROM chat_history")
    return cursor.fetchall()


def clear_chat_history():
    cursor.execute("DELETE FROM chat_history")
    conn.commit()


# UI using streamlt

st.title("Groq RAG Chatbot")



if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None


with st.sidebar:
    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        clear_chat_history()
        st.rerun()



for role, message in st.session_state.messages:
    with st.chat_message(role):
        st.write(message)



uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    loader = PyPDFLoader(temp_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)
    st.session_state.vectorstore = vectorstore

    st.success("Document processed and ready for RAG!")



# LLM Setup

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="llama-3.1-8b-instant"
)

prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the provided context.

Context:
{context}

Question:
{question}
""")



# Chat Input

user_input = st.chat_input("Ask something...")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)

    save_message("user", user_input)
    st.session_state.messages.append(("user", user_input))

    if st.session_state.vectorstore:
        retriever = st.session_state.vectorstore.as_retriever()

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        answer = rag_chain.invoke(user_input)

    else:
        answer = llm.invoke(user_input).content

    with st.chat_message("assistant"):
        st.write(answer)

    save_message("assistant", answer)
    st.session_state.messages.append(("assistant", answer))
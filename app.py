import streamlit as st
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


from dotenv import load_dotenv
load_dotenv()

st.title("RAG Application build on Gemini Model")

loader = PyPDFLoader("Python.pdf")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000)
docs = text_splitter.split_documents(data)

vectorstore = Chroma.from_documents(documents=docs, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))

retreiver = vectorstore.as_retriever(search_type = "similarity", seach_kwargs={"k": 10})

llm = ChatGoogleGenerativeAI(model = "gemini-1.5-pro", temperature = 0.3, max_tokens= 500)

query = st.chat_input("Ask something: ")
prompt = query

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retreived context to answer the question"
    "If you dont know the answer, say Thank you, I don't know "
    "Use three sentences maximum and make the answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system_prompt),
        ("human","{input}")
    ]
)

if query:
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retreiver, question_answer_chain)

    response = rag_chain.invoke({"input": query})
    st.write(response["answer"])


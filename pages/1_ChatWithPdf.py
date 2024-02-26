import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.callbacks import get_openai_callback
import os
import subprocess
import sys

def install(name):
    subprocess.call([sys.executable, '-m', 'pip', 'install', name])

install("faiss-cpu")

with st.sidebar:
    st.title("PDF At Sor")
    st.markdown("""
    ## Hakkında
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models)""")
    st.write("Made by emirg")

def main():
    st.header("PDF ile Soru Sor")

    pdf = st.file_uploader("", type = "pdf")
    
    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
        )
        chunks = text_splitter.split_text(text = text)

        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embeddings)
        query = st.text_input("Soru sor.")

        if query:
            docs = VectorStore.similarity_search(query, k = 3)
            llm = OpenAI()
            chain = load_qa_chain(llm = llm, chain_type = "stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents = docs, question = query)
                print(cb)
            st.write(response)

if __name__ == "__main__":
    main()
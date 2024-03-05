from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import CohereEmbeddings
import chromadb
import os
import toml

def ApiKeyConf(secret_path, api_name):
    with open(secret_path, "r") as f:
        config = toml.load(f)

    return config.get(api_name), config.get("COHERE_API_KEY")

OPENAI_API_KEY, COHERE_API_KEY = ApiKeyConf("./.streamlit/secrets.toml", "OPENAI_API_KEY")

dataset_file = "./datasets"
documents = []

def load_chunk_persist_pdf():
    for file in os.listdir(dataset_file):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(dataset_file, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())

        text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
        chunks = text_splitter.split_documents(documents)
        vectordb = Chroma.from_documents(
            documents = chunks,
            embedding = CohereEmbeddings(cohere_api_key = COHERE_API_KEY),
        )
        vectordb.persist()
        return vectordb
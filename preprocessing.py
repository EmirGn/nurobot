from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import chromadb
import os
import toml

def ApiKeyConf(secret_path, api_name):
    with open(secret_path, "r") as f:
        config = toml.load(f)

    return config.get(api_name) if config else print("{api_name} not found.")

OPENAI_API_KEY = ApiKeyConf("./.streamlit/secrets.toml", "OPENAI_API_KEY")

dataset_file = "./datasets"
documents = []

def load_chunk_persist_pdf():
    for file in os.listdir(dataset_file):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(dataset_file, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())

        text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 10)
        chunks = text_splitter.split_documents(documents)
        client = chromadb.Client()
        if client.list_collections():
            consent_collection = client.create_collection("consent_collection")
        else:
            print("Collection already exists.")
        vectordb = Chroma.from_documents(
            documents = chunks,
            embedding = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY),
            persist_directory = "./vectorss"
        )
        vectordb.persist()
        return vectordb
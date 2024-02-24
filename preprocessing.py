# %%
import os
import toml

# %%
import requests

if os.path.exists("/Users/emirg/Nodes/muhammed/datasets/teknofest_scheme.pdf"):
    print("The target file exists in the directory.")
else:
    url = "https://cdn.teknofest.org/media/upload/userFormUpload/TEKNOFEST_2024_Roket_Yar%C4%B1smas%C4%B1_Sartnamesi_Ver2.4_NltT9.pdf"
    name = "teknofest_scheme"
    response = requests.get(url)
    with open(name, "w") as f:
        f.write(response.text)

# %%
with open("./.streamlit/secrets.toml", "r") as f:
    config = toml.load(f)

OPENAI_API_KEY = config.get("OPENAI_API_KEY")
if OPENAI_API_KEY:
    print("Your API Key is available")
else:
    print("API key not found")

# %%
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import chromadb

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
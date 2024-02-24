# %%
import dotenv
import os

OPENAI_API_KEY = "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"

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
name1 = "./datasets/teknofest_scheme.pdf"

# %%
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader("./datasets")
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500)
chunks = text_splitter.split_documents(pages)

# %%
from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)

# %%
import weaviate
from langchain.vectorstores import Weaviate
from weaviate.embedded import EmbeddedOptions

client = weaviate.Client(
    url = "https://muhammed-75p0dubq.weaviate.network",
    additional_headers = {"openai-key": OPENAI_API_KEY},
    startup_period = 10,
)

# %%
vector_db = Weaviate.from_documents(
    client = client, documents = chunks, embedding = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY), by_text = False
)
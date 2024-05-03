from preprocessing import *

from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

gpt_models = ["gpt-4-1106-preview", "gpt-3.5-turbo"]
llm = ChatGroq(model_name = "llama3-70b-8192", groq_api_key = GROQ_API_KEY)

template = """ Soruları contexte göre cevapla. {context}
YAPMAN GEREKENLER:
-Soruları contexte göre cevaplamak.
-Eğer promptun sonunda "-free" yazıyorsa normal bir chatbot gibi davran bütün sorular
Question: {question}
"""

vectordb = load_chunk_persist_pdf()
prompt = ChatPromptTemplate.from_template(template)
retriever = vectordb.as_retriever()
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
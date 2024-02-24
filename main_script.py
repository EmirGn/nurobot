# %%
from preprocessing import *

# %%
from langchain.chat_models import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate

llm = ChatOpenAI(model_name = "gpt-3.5-turbo", openai_api_key = OPENAI_API_KEY)
template = """Answer the question based only on the following context: {context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# %%
retriever = vector_db.as_retriever()

# %%
from langchain.schema.output_parser import StrOutputParser

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
# %% [markdown]
# #**INTERFACE**

# %% [markdown]
# 



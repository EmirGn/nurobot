# %%
from preprocessing import *

# %%
from langchain.chat_models import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

llm = ChatOpenAI(model_name = "gpt-4-1106-preview", openai_api_key = OPENAI_API_KEY)
template = """ Soruları contexte göre cevapla. {context}
YAPMAN GEREKENLER:
-Sana verilen promptda herhangi bir flag yoksa "-free, -matematik" cevabını sadece raporlar göre ver.
Eğer raporlarda cevap yoksa cevabı bulamadığın söyle.
-Sana verilen promptda -free işaretini görürsen normal bir chatbot gibi davran ve raporlardan bağımsız cevap ver.
Cevabını detaylı bir şekilde ver.
-Sana verilen promptda -matematik işaretini görürsen kullanıcıya matematiksel işlemlerde yardım et ve gerektiğinde rapordan bağımsız cevap ver.
-Cevaplarını detaylı vermek ZORUNDASIN.
-Flag/işaret almadıkça raporlara bağlı kalıp cevap vermek zorundasın.
Question: {question}
"""

vectordb = load_chunk_persist_pdf()
prompt = ChatPromptTemplate.from_template(template)
retriever = vectordb.as_retriever()
# %%
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



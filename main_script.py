# %%
from preprocessing import *

# %%
from langchain.chat_models import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

llm = ChatOpenAI(model_name = "gpt-4", openai_api_key = OPENAI_API_KEY)
template = """Soruları CONTEXTe göre cevapla ve cevapların tek ANA kısımdan oluşacak:
- Konu Hakkında Direkt Bilgi:
    UZUN VE DETAYLI CEVAPLAR VER.
    Sana sorulan sorulara verebildiğin kadar detaylı cevaplar ver. Verilen tüm bilgileri kullan ve istenilen herşeyi ver. 
    Şartname ile ilgili sorulara kesinlikle doğru cevap ver ve sayısal verilerle cevap vermeye çalış. Kaynak belirt.
    Örnek raporla ilgili sorulara detaylıca cevap ver ve sayısal verilerin rapordakiyle aynı olduğundan emin ol.
    Aynı verinin raporda farklı şekillerde kullanıldığını görürsen (roket yarıçap: 1 veya 2) bu sayıların nerelerde kullanıldığını yaz ve kaynak belirt.
    Matematiksel işlemlere yardım et. Roket yapıyoruz.

    {context}
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



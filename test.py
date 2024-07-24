import os
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from operator import itemgetter
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
response = model.invoke("How are you")
print(response)
template = """
Answer the question based on the context below. If you can't 
answer the question, reply "I don't know".

Context: {context}

Question: {question}
"""
parser = StrOutputParser()
prompt = ChatPromptTemplate.from_template(template)
prompt.format(context="Mary's sister is Susana", question="Who is Mary's sister?")
chain = prompt | model | parser
translation_prompt = ChatPromptTemplate.from_template(
    "Translate {answer} to {language}"
)
translation_chain = (
    {"answer": chain, "language": itemgetter("language")} | translation_prompt | model | parser
)

loader = TextLoader("transcription.txt")
text_documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
chunks = text_splitter.split_documents(text_documents)
embeddings = OpenAIEmbeddings()
soru = str(input("what do you wanna know about mulholland drive?"))
embedded_query = embeddings.embed_query(soru)
vectorstore1 = DocArrayInMemorySearch.from_texts(
    texts=[chunk.page_content for chunk in chunks],
    embedding=embeddings,
)
retriever1 = vectorstore1.as_retriever()
chunkss = retriever1.invoke(soru)
setup = RunnableParallel(context=retriever1, question=RunnablePassthrough())
setup.invoke(soru)
chain = setup | prompt | model | parser
cevap = chain.invoke(soru)
print(cevap)

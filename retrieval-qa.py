from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader("./sample-input.txt", encoding="utf-8")

load_dotenv()

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
store = Chroma.from_documents(texts, embeddings,collection_name="sample_collection")

llm = ChatOpenAI(temperature=0.2)
chain = RetrievalQA.from_chain_type(llm,retriever=store.as_retriever())

question = input("Enter your question: ")

answer= chain.invoke(question)

print(f"Answer: {answer['result']}")
# First install required packages
#!pip install azure-identity langchain-openai chromadb==0.5.5 langchain==0.2.11 
#!pip install langchain-community==0.2.10 langchain-text-splitters==0.2.2
#!pip install unstructured==0.15.0 unstructured[pdf]==0.15.0
#!apt-get install poppler-utils tesseract-ocr libtesseract-dev

import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import AzureChatOpenAI  # Changed from Groq

# Azure OpenAI Configuration
os.environ["AZURE_OPENAI_API_KEY"] = "<AZURE_AI_KEY>"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://<resource>.openai.azure.com/"
os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = "gpt-4-deployment"  # deployment name

# Fetch and process PDF (unchanged)
url = "https://dspmuranchi.ac.in/pdf/Blog/Python%20Built-In%20Functions.pdf"   # we can use input()
response = requests.get(url)
with open("python_inbuildfunction.pdf", "wb") as f:
    f.write(response.content)

loader = PyPDFLoader("/content/python_inbuildfunction.pdf")
documents = loader.load()

text_splitter = CharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=400
)
texts = text_splitter.split_documents(documents)

# Vector DB setup (unchanged)
embeddings = HuggingFaceEmbeddings()
persist_directory = "vector_db"
vectordb = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory=persist_directory
)
retriever = vectordb.as_retriever()

# Azure OpenAI GPT-4 initialization
llm = AzureChatOpenAI(
    deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    temperature=0,
    api_version="2024-02-15-preview"  # Use latest stable version
)

# QA Chain (unchanged)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Query execution (unchanged)
query = input()
response = qa_chain.invoke({"query": query})
print(response["result"])
print("*"*30)
print("Source Document:", response["source_documents"][0].metadata["source"])

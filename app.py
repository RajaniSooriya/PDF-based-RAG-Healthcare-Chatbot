from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os


# Folder containing all PDFs
pdf_folder = "data"

# Step 1
# Load all PDFs from the folder named 'data'

docs = []
for pdf_file in os.listdir(pdf_folder):
    if pdf_file.endswith(".pdf"):
        loader =PyPDFLoader(os.path.join(pdf_folder,pdf_file))
        docs.extend(loader.load())

print("pdfs loaded")

# Step 2 
# Split texts into small chunks


text_splitter =  RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
split_docs = text_splitter.split_documents(docs)

print(f"loaded{len(split_docs)}chunks from {len(docs)}pdf documents.")
      

# Step 3
# Creating a vector database by converting the text chunks into embeddings

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(split_docs, embeddings)

print("vector store created")

# Step 4 
# Creating the LLM and RAG Chain

from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature = 0.3)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever(search_kwargs={"k": 3}))


# Step 5
# Ask Questions

query = "What will the rounding practioner be responsible for?"
response = qa_chain.invoke(query)
print("Answer:")
print(response)

from langchain.document_loaders import PyPDFLoader
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
      
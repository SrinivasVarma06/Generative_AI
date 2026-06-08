# Write your code here
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

file_path = "./attention_all_you_need.pdf"
loader = PyPDFLoader(file_path)
documents=loader.load()
text_splitter = RecursiveCharacterTextSplitter()
split_documents = text_splitter.split_documents(documents)

print(split_documents)
print(len(split_documents))
print(split_documents[0].metadata)
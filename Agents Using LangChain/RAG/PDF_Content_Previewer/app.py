# Write your code here
from langchain_community.document_loaders import PyPDFLoader
file_path = "./attention_all_you_need.pdf"
loader = PyPDFLoader(file_path)
document=loader.load()
print(document[0].metadata)
print(document[0].page_content)
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader("./tool_calling.py")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)
print(len(chunks))
print(chunks[0].page_content)
print(chunks[1].page_content)
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

model = init_chat_model(
    "google_genai:gemini-2.5-flash",
    api_key=api_key
)

loader = WebBaseLoader(
    [
        "https://docs.langchain.com/oss/python/integrations/document_loaders",
        "https://docs.langchain.com/oss/python/integrations/vectorstores",
        "https://docs.langchain.com/oss/python/integrations/text_embedding",
    ]
)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

all_splits = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db"
)

vector_store.add_documents(documents=all_splits)

sample_embedding = vector_store.get(
    limit=1,
    include=["embeddings", "documents", "metadatas"]
)

print(sample_embedding)


def retrieve_context(query):
    retrieved_docs = vector_store.similarity_search(query, k=3)

    context_parts = []

    for doc in retrieved_docs:
        context_parts.append(
            f"Source: {doc.metadata}\nContent: {doc.page_content}"
        )

    context = "\n\n".join(context_parts)

    return context, retrieved_docs


def ask_about_pdf(user_query):
    context, source_docs = retrieve_context(user_query)

    messages = [
        {
            "role": "system",
            "content": "Answer strictly based on the retrieved documentation context."
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{user_query}"
        }
    ]

    response = model.invoke(messages)

    return {
        "answer": response.content,
        "source_documents": source_docs,
        "context_used": context
    }


result1 = ask_about_pdf(
    "How to use the HuggingFaceEmbeddings?"
)
print(result1["answer"])

result2 = ask_about_pdf(
    "Explain the use this value : sentence-transformers/all-mpnet-base-v2"
)
print(result2["answer"])

result3 = ask_about_pdf(
    "How to use Open AI Embeddings"
)
print(result3["answer"])

result4 = ask_about_pdf(
    "Explain about this: OpenAIEmbeddings"
)
print(result4["answer"])
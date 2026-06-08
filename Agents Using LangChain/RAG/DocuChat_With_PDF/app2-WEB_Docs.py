import os
from dotenv import load_dotenv

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Gemini model
model = init_chat_model(
    "google_genai:gemini-2.5-flash",
    api_key=GEMINI_API_KEY
)

# Load webpages
urls = [
    "https://docs.langchain.com/oss/python/integrations/document_loaders",
    "https://docs.langchain.com/oss/python/integrations/vectorstores",
    "https://docs.langchain.com/oss/python/integrations/text_embedding",
]

loader = WebBaseLoader(urls)
docs = loader.load()

# Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)

all_splits = text_splitter.split_documents(docs)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# Create Chroma vector store
vector_store = Chroma(
    collection_name="example_collection",
    persist_directory="./chroma_langchain_db",
    embedding_function=embeddings,
)

# Add documents to vector store
vector_store.add_documents(all_splits)

# Retrieve sample embedding
sample_embedding = vector_store.get(
    limit=1,
    include=["embeddings", "documents", "metadatas"]
)

print(sample_embedding)


def retrieve_context(query):
    retrieved_docs = vector_store.similarity_search(
        query,
        k=3
    )

    context_parts = []

    for doc in retrieved_docs:
        context_parts.append(
            f"Source: {doc.metadata}\n"
            f"Content: {doc.page_content}"
        )

    context = "\n\n".join(context_parts)

    return context, retrieved_docs


def ask_about_pdf(user_query):
    context, source_docs = retrieve_context(user_query)

    messages = [
        {
            "role": "system",
            "content": (
                "Answer strictly based on the retrieved documentation context. "
                "If the answer is not present in the context, say so."
            ),
        },
        {
            "role": "user",
            "content": f"""
            Context:
            {context}

            Question:
            {user_query}
            """,
        },
    ]

    response = model.invoke(messages)

    return {
        "answer": response.content,
        "source_documents": source_docs,
        "context_used": context,
    }


queries = [
    "How to use the HuggingFaceEmbeddings?",
    "Explain the use this value : sentence-transformers/all-mpnet-base-v2",
    "How to use Open AI Embeddings",
    "Explain about this: OpenAIEmbeddings",
]

for query in queries:
    result = ask_about_pdf(query)
    print(result["answer"])
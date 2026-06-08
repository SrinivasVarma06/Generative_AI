from dotenv import load_dotenv
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
load_dotenv()

loader = PyPDFLoader("Attention_Is_All_You_Need.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
    collection_name="attention_paper",
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
vector_store.add_documents(chunks)
sample_data = vector_store.get()
print(sample_data)

model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0
)

def retrieve_context(query):
    docs = vector_store.similarity_search(query, k=3)

    context_parts = []
    for doc in docs:
        context_parts.append(
            f"Source: {doc.metadata}\nContent: {doc.page_content}"
        )

    context = "\n\n".join(context_parts)
    return context, docs


def ask_about_pdf(user_query):
    context, source_docs = retrieve_context(user_query)
    system_message = SystemMessage(
        content="Answer questions using only the provided context."
    )

    user_message = HumanMessage(
        content=f"""
        Context:
        {context}

        Question:
        {user_query}
        """
    )

    response = model.invoke(
        [system_message, user_message]
    )

    return {
        "answer": response.content,
        "source_documents": source_docs,
        "context_used": context
    }


result = ask_about_pdf(
    "What is the main contribution of the Transformer paper?"
)

print(result)                    
print(result["answer"])

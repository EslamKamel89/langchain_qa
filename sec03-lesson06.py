import shutil
import tempfile

import numpy as np
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-small")

SAMPLE_DOCS = [
    Document(
        page_content="Vector stores are databases optimized for storing and searching embeddings.",
        metadata={"source": "vector_guide", "topic": "database"},
    ),
    Document(
        page_content="RAG combines retrieval with generation for more accurate LLM responses.",
        metadata={"source": "rag_guide", "topic": "architecture"},
    ),
    Document(
        page_content="Embeddings convert text into numerical vectors for semantic similarity.",
        metadata={"source": "embeddings_guide", "topic": "fundamentals"},
    ),
    Document(
        page_content="Chroma is an open-source embedding database for AI applications.",
        metadata={"source": "chroma_docs", "topic": "database"},
    ),
    Document(
        page_content="FAISS is a library for efficient similarity search developed by Facebook.",
        metadata={"source": "faiss_docs", "topic": "database"},
    ),
    Document(
        page_content="Pinecone is a managed vector database service for production workloads.",
        metadata={"source": "pinecone_docs", "topic": "database"},
    ),
    Document(
        page_content="LangChain is a framework for developing applications powered by language models.",
        metadata={"source": "langchain_docs", "topic": "overview"},
    ),
    Document(
        page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs.",
        metadata={"source": "langgraph_docs", "topic": "overview"},
    ),
]


def chroma_basics():
    vector_store = Chroma.from_documents(
        documents=SAMPLE_DOCS,
        embedding=embedding,
        persist_directory="./chroma_db",
    )
    print(f"Vector store created {vector_store._collection.count()} and persisted")
    query = "What's langchain"
    results = vector_store.similarity_search(query, k=2)
    print(f"Top two results for the query: {query}")
    for result in results:
        print("-------------------------")
        print(result.page_content)
        print(result.metadata)


def similarity_search_with_scores():
    vector_store = Chroma.from_documents(
        documents=SAMPLE_DOCS,
        embedding=embedding,
        persist_directory="./chroma_db",
    )
    query = "Explain vector stores"
    results_with_scores = vector_store.similarity_search_with_score(query, k=3)
    for doc, score in results_with_scores:
        print(score, ":", doc.page_content)


def similarity_search_with_filters():
    vector_store = Chroma.from_documents(
        documents=SAMPLE_DOCS,
        embedding=embedding,
        persist_directory="./chroma_db",
    )
    query = "What's a database"
    results = vector_store.similarity_search(
        query=query, k=2, filter={"topic": "database"}
    )
    for res in results:
        print("----------------------------")
        print(res.page_content)


if __name__ == "__main__":
    # chroma_basics()
    # similarity_search_with_scores()
    similarity_search_with_filters()

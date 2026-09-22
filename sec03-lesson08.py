import hashlib
import shutil
import tempfile
import time

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")


def create_retriever(
    texts: list[str],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    k: int = 3,
):
    docs = [Document(page_content=text) for text in texts]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    split_docs = splitter.split_documents(docs)

    vector_store = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding_model,
        persist_directory="./chromadb",
    )

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


SAMPLE_TEXTS = [
    """
    Python is a general-purpose programming language widely used for backend
    development, automation, data science, and artificial intelligence.
    Django and FastAPI are popular Python frameworks for building web APIs.
    """,
    """
    JavaScript is the primary programming language of web browsers.
    It is commonly used for interactive frontend applications, while Node.js
    allows JavaScript to run on servers for backend development.
    """,
    """
    PostgreSQL is a relational database management system. It supports SQL,
    transactions, indexes, constraints, and advanced data types. Applications
    commonly use PostgreSQL when strong relational data integrity is required.
    """,
    """
    Redis is an in-memory data store commonly used for caching, message
    brokering, sessions, and other workloads requiring very fast access.
    Keeping frequently accessed data in Redis can reduce database load.
    """,
    """
    LangChain is a framework for building applications powered by language
    models. It provides abstractions for models, prompts, document loaders,
    text splitters, embeddings, vector stores, retrievers, and chains.
    """,
    """
    Vector stores store embeddings and support semantic similarity search.
    Documents are converted into numerical vectors by an embedding model.
    A query can then be embedded and compared with stored vectors to retrieve
    semantically relevant documents.
    """,
]

QUERIES = [
    "Which Python frameworks can I use to build backend APIs?",
    "What database should I use when relational integrity is important?",
    "How can I reduce database load using an in-memory cache?",
    "How does semantic document retrieval work?",
    "What does LangChain provide for building LLM applications?",
]

retriever = create_retriever(
    SAMPLE_TEXTS,
    chunk_size=200,
    chunk_overlap=20,
    k=2,
)

for query in QUERIES:
    results = retriever.invoke(query)

    print(f"\nResults for query: {query}")

    for doc in results:
        print(doc.page_content)
        print("---")

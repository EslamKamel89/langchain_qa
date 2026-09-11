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

PERSIST_DIRECTORY = "./chromadb"
COLLECTION_NAME = "langchain_course"

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
)

documents = [
    Document(
        page_content="LangChain is a framework for developing applications powered by language models.",
        metadata={"source": "LangChain Docs", "topic": "framework"},
    ),
    Document(
        page_content="LangGraph is used for building stateful agent workflows.",
        metadata={"source": "LangGraph Docs", "topic": "framework"},
    ),
    Document(
        page_content="Vector stores are databases optimized for storing and searching embeddings.",
        metadata={"source": "Vector Store Guide", "topic": "database"},
    ),
    Document(
        page_content="Chroma is an open-source embedding database for AI applications.",
        metadata={"source": "Chroma Docs", "topic": "database"},
    ),
    Document(
        page_content="Pinecone is a managed vector database.",
        metadata={"source": "Pinecone Docs", "topic": "database"},
    ),
    Document(
        page_content="Embeddings convert text into numerical vectors for semantic similarity.",
        metadata={"source": "Embeddings Guide", "topic": "fundamentals"},
    ),
]


def document_id(document: Document):
    content = document.page_content + document.metadata.get("source", "")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def get_vector_store() -> Chroma:
    return Chroma(
        persist_directory=PERSIST_DIRECTORY,
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
    )


def add_missing_documents(store: Chroma, documents: list[Document]):
    ids = [document_id(doc) for doc in documents]
    existing = store.get(ids=ids)
    existing_ids = set(existing["ids"])
    missing_documents = []
    missing_ids = []
    for doc, doc_id in zip(documents, ids):
        if doc_id not in existing_ids:
            missing_documents.append(doc)
            missing_ids.append(doc_id)
    if not missing_documents:
        print("No missing documents.")
        return
    print(f"Adding {len(missing_documents)} missing documents.")
    store.add_documents(missing_documents, ids=missing_ids)
    print(f"Added {len(missing_documents)} missing documents.")


vector_store = get_vector_store()
add_missing_documents(vector_store, documents)

print(f"Vector store count: {vector_store._collection.count()}")


def run_search_loop(chroma: Chroma):
    print("\nInteractive search started.")
    print("Press Ctrl+C to exit.\n")
    try:
        while True:
            query = input("Enter your search query: ")
            if not query:
                continue
            results = chroma.similarity_search_with_score(query, k=3)
            print()
            for index, (result, score) in enumerate(results, start=1):
                print("-" * 40)
                print(f"Result {index}: {result.page_content}")
                print(f"Metadata: {result.metadata}")
                print(f"Score: {score}")
            print()

    except KeyboardInterrupt:
        print("\nExiting search loop.")


if __name__ == "__main__":
    run_search_loop(vector_store)

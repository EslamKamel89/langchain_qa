from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from advanced_rag_lab.knowledge_base import create_vector_store, show_docs

LONG_DOCUMENT = Document(
    page_content=(
        "LangGraph Overview. "
        "LangGraph is a framework for building stateful AI agents. "
        "It represents workflows using nodes, edges, and shared state. "
        "Nodes execute application logic and can call language models "
        "or external tools. "
        "Edges control transitions between workflow steps. "
        "State carries information across the execution graph. "
        "Conditional edges support branching decisions. "
        "Cycles allow agents to repeat steps when necessary. "
        "LangGraph is useful for multi-step workflows that require "
        "state management and explicit execution control. "
        "It can coordinate tool use, reasoning steps, and "
        "human review within an agent workflow. "
        "LangChain provides model, prompt, and retrieval abstractions "
        "that can be used within LangGraph applications."
    ),
    metadata={"source": "langgraph-guide.md"},
)


def main():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(
        collection_name="parent_child_lab",
        embedding_function=embeddings,
        persist_directory="./chromadb",
    )
    doc_store = InMemoryStore()
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
    )
    retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=doc_store,
        parent_splitter=parent_splitter,
        child_splitter=child_splitter,
        search_kwargs={"k": 2},
    )
    query = "How does LangGraph manage workflow state?"
    retriever.invoke(query)
    print("Matched child chunks:")
    show_docs(vector_store.as_retriever(search_kwargs={"k": 2}).invoke(query))

    print("Returned parent documents:")
    show_docs(retriever.invoke(query))


if __name__ == "__main__":
    main()

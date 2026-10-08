from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
documents = [
    Document(
        page_content=(
            "LangChain is a framework for building applications powered "
            "by language models. It provides abstractions for models, "
            "prompts, retrieval, tools, and chains."
        ),
        metadata={"source": "langchain.md", "id": 1},
    ),
    Document(
        page_content=(
            "LangGraph is designed for building stateful workflows and "
            "agents. Applications are modeled as graphs containing nodes, "
            "edges, state, and control flow."
        ),
        metadata={"source": "langgraph.md", "id": 2},
    ),
    Document(
        page_content=(
            "Retrieval-Augmented Generation retrieves relevant external "
            "information and provides it to a language model as context "
            "before generation."
        ),
        metadata={"source": "rag.md", "id": 3},
    ),
]

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = []
chunk_ids = []
for document in documents:
    document_chunks = splitter.split_documents([document])
    for index, chunk in enumerate(document_chunks):
        chunks.append(chunk)
        chunk_ids.append(f"{document.metadata['id']}:{index}")


embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = Chroma(
    collection_name="structured_rag_lab",
    embedding_function=embeddings,
    persist_directory="./chromadb",
)

existing = vector_store.get(ids=chunk_ids)
existing_ids = set(existing["ids"])
new_chunks = []
new_ids = []

for chunk, chunk_id in zip(chunks, chunk_ids):
    if chunk_id not in existing_ids:
        new_chunks.append(chunk)
        new_ids.append(chunk_id)

if new_chunks:
    vector_store.add_documents(new_chunks, ids=new_ids)


print(f"Added {len(new_chunks)} new chunks to the vector store.")

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})

retrieved_docs = retriever.invoke("What is LangGraph used for?")
for doc in retrieved_docs:
    print("-" * 20)
    print(doc.page_content)
    print(doc.metadata)
    print()


def format_documents_with_source(docs: list[Document]):
    sections = []
    for doc in documents:
        source = doc.metadata.get("source", "unknown")

        sections.append(f"Source: {source}\n" f"Content: {doc.page_content}")
    return "\n\n".join(sections)

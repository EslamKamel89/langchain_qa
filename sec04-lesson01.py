from operator import itemgetter

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

documents = [
    Document(
        page_content="""
        LangChain is a framework for building applications powered by
        language models. It provides abstractions for models, prompts,
        retrieval, tools, and chains.
        """,
        metadata={"source": "langchain.md"},
    ),
    Document(
        page_content="""
        LangGraph is designed for building stateful workflows and agents.
        Applications are modeled as graphs containing nodes, edges,
        state, and control flow.
        """,
        metadata={"source": "langgraph.md"},
    ),
    Document(
        page_content="""
        Retrieval-Augmented Generation retrieves relevant external
        information and provides it to a language model as context
        before generation.
        """,
        metadata={"source": "rag.md"},
    ),
]


def main():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(documents)
    # for index, chunk in enumerate(chunks):
    #     print(f"Chunk {index}:")
    #     print(chunk.page_content)
    #     print("---")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chromadb",
    )
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 2}
    )
    # docs = retriever.invoke("What's langgraph is used for ?")
    # print(format_docs_with_sources(docs))
    # for doc in docs:
    #     print(f"Source: {doc.metadata['source']}")
    #     print(doc.page_content)
    prompt = ChatPromptTemplate.from_template("""
You are answering questions using a private knowledge base.

Rules:
1. Answer only from the supplied context.
2. Do not use outside knowledge.
3. If the context does not contain enough information, say:
   "I don't have information about that in my knowledge base."
4. Mention the source used for the answer.
5. Keep the answer concise.

Context:
{context}

Question:
{question}

Answer:
    
    """)
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    parser = StrOutputParser()
    chain = (
        {
            "context": itemgetter("question") | retriever | format_docs_with_sources,
            "question": itemgetter("question") | RunnablePassthrough(),
        }
        | prompt
        | llm
        | parser
    )
    questions = [
        "What is LangGraph used for?",
        "What does RAG do?",
        "What components does LangChain provide?",
        "Who invented PostgreSQL?",
    ]
    for question in questions:
        print(f"Question: {question}")
        print(chain.invoke({"question": question}))
        print("---")


def format_docs_with_sources(docs: list[Document]):
    sections = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        sections.append(f"Source: {source}\nContent: {doc.page_content}")
    return "\n\n".join(sections)


if __name__ == "__main__":
    main()

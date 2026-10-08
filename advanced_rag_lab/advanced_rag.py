from langchain_classic.retrievers import (
    ContextualCompressionRetriever,
    MultiQueryRetriever,
)
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from advanced_rag_lab.knowledge_base import TECH_DOCS, create_llm, create_vector_store


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        (f"Source: {doc.metadata.get('source', 'unknown')}\n" f"{doc.page_content}")
        for doc in docs
    )


def main():
    vector_store = create_vector_store(TECH_DOCS, "advanced_rag_lab")
    llm = create_llm()
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever, llm=llm, include_original=True
    )
    compressor = LLMChainExtractor.from_llm(llm=llm)
    advanced_retriever = ContextualCompressionRetriever(
        base_retriever=multi_query_retriever, base_compressor=compressor
    )
    prompt = ChatPromptTemplate.from_template("""
Answer the question using only the retrieved context.
If the context is insufficient, say you don't know.

Context:
{context}

Question:
{question}

Answer:
""")
    rag_chain = (
        {"context": advanced_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    questions = [
        "What options do I have for building AI agents?",
        "How can I store and search embeddings?",
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        print(f"Answer: {rag_chain.invoke(question)}")


if __name__ == "__main__":
    main()

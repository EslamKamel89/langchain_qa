from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from advanced_rag_lab.knowledge_base import (
    TECH_DOCS,
    create_llm,
    create_vector_store,
    show_docs,
)


def main():
    vector_store = create_vector_store(TECH_DOCS, "hybrid_lab")
    keyword_retriever = BM25Retriever.from_documents(TECH_DOCS)
    keyword_retriever.k = 3
    semantic_retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    hybrid_retriever = EnsembleRetriever(
        retrievers=[keyword_retriever, semantic_retriever],
        weights=[0.4, 0.6],
        id_key="source",
    )
    queries = [
        "Which database supports ACID transactions and pgvector?",
        "What tools help build stateful AI agents?",
        "How can I store embeddings for similarity search?",
    ]
    for query in queries:
        print(f"\nQUERY: {query}")

        print("\nBM25:")
        show_docs(keyword_retriever.invoke(query)[:1])

        print("Semantic:")
        show_docs(semantic_retriever.invoke(query)[:1])

        print("Hybrid:")
        show_docs(hybrid_retriever.invoke(query)[:3])


if __name__ == "__main__":
    main()

import logging

from langchain_classic.retrievers import MultiQueryRetriever

from advanced_rag_lab.knowledge_base import (
    TECH_DOCS,
    create_llm,
    create_vector_store,
    show_docs,
)

logging.basicConfig(level=logging.INFO)
logging.getLogger("langchain_classic.retrievers.multi_query").setLevel(logging.INFO)


def main() -> None:
    vector_store = create_vector_store(TECH_DOCS, "multi_query_lab")
    base_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2},
    )
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=create_llm(),
        include_original=True,
    )
    query = "What tools can I use to build AI applications?"
    print("-" * 20)
    print("Single Query Results")
    show_docs(base_retriever.invoke(query))
    print("-" * 20)
    print("Multi Query Results")
    show_docs(multi_query_retriever.invoke(query))


if __name__ == "__main__":
    main()

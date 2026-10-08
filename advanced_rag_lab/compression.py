from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document

from advanced_rag_lab.knowledge_base import create_llm, create_vector_store, show_docs

LONG_DOCS = [
    Document(
        page_content=(
            "Acme Solutions was founded in 2010. "
            "The company operates offices in several cities. "
            "Its employees organize community events and maintain "
            "internal business reporting systems. "
            "The company uses AWS for hosting and PostgreSQL "
            "for transactional application data. "
            "LangChain provides components for building LLM "
            "applications, including prompts, retrievers, "
            "tools, and model integrations. "
            "The company also publishes an annual financial report. "
            "Its engineering team uses LangGraph to build "
            "stateful agent workflows with nodes, edges, "
            "and shared execution state. "
            "The team maintains several internal dashboards "
            "for monitoring infrastructure and expenses."
        ),
        metadata={"source": "acme-engineering.md"},
    ),
    Document(
        page_content=(
            "Northstar Software builds business applications. "
            "It operates customer support and billing systems. "
            "The engineering team uses LangChain to connect "
            "LLMs with application tools and retrieval systems. "
            "LangGraph helps coordinate multi-step, stateful "
            "agent workflows. "
            "The company also manages office equipment, "
            "staff scheduling, and customer billing."
        ),
        metadata={"source": "northstar.md"},
    ),
]


def main() -> None:
    vector_store = create_vector_store(LONG_DOCS, "compression_lab")
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    compressor = LLMChainExtractor.from_llm(create_llm())
    compression_retriever = ContextualCompressionRetriever(
        base_retriever=base_retriever,
        base_compressor=compressor,
    )
    query = "What frameworks are used to build LLM applications?"
    original_docs = base_retriever.invoke(query)
    print("-" * 80)
    print("Original Docs:")
    show_docs(original_docs)
    print("-" * 80)
    compressed_docs = compression_retriever.invoke(query)
    print("Compressed Docs:")
    show_docs(compressed_docs)
    print("-" * 80)


if __name__ == "__main__":
    main()

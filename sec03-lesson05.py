import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()


embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def basic_embeddings():
    text = "What's machine learning"
    single_embedding = embeddings.embed_query(text)
    print(f"Vector dimensions: {len(single_embedding)}")
    print(f"first five values: {single_embedding[:5]}")
    print(f"Vector norm: {np.linalg.norm(single_embedding)}")


def batch_embeddings():
    texts = [
        "What's machine learning",
        "Explain the concept of overfitting in ML.",
        "how does a neural network work?.",
    ]
    embeds = embeddings.embed_documents(texts)
    for i, embed in enumerate(embeds):
        print(f" ---------------------- {i} ---------------------- ")
        print(f"Vector dimensions: {len(embed)}")
        print(f"first five values: {embed[:5]}")
        print(f"Vector norm: {np.linalg.norm(embed)}")


def similarity_search():
    docs = [
        "Cats are popular pets",
        "Machine learning enables AI applications",
        "Deep learning uses neural networks",
        "Python is a programming language",
        "javascript is used for web development",
    ]
    query = "What programming language exist?"
    docs_vector = embeddings.embed_documents(docs)
    query_vector = embeddings.embed_query(query)

    def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    similarities = [cosine_similarity(doc_vec, query_vector) for doc_vec in docs_vector]
    ranked_docs = sorted(zip(docs, similarities), key=lambda x: x[1], reverse=True)
    for doc, score in ranked_docs:
        print("------------------------------")
        print(score, " : ", doc)


def embed_caching():
    import tempfile

    from langchain_classic.embeddings.cache import CacheBackedEmbeddings
    from langchain_classic.storage import LocalFileStore

    with tempfile.TemporaryDirectory() as tempdir:
        store = LocalFileStore(root_path=tempdir)
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings=embeddings,
            document_embedding_cache=store,
            namespace="exercise",
        )
        text = "What's Reinforcement learning?"
        print("First api call")
        vectors1 = cached_embeddings.embed_query(text)
        print(len(vectors1))
        print("Second call from cache")
        vectors2 = cached_embeddings.embed_query(text)
        print(len(vectors2))
        print(f"\nSave vectors: {np.allclose(vectors1 , vectors2)}")


if __name__ == "__main__":
    # basic_embeddings()
    # batch_embeddings()
    # similarity_search()
    embed_caching()

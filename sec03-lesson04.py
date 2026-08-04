from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def embeddings_demo():
    text = "This is sample text to be embedded"
    embedding = embeddings.embed_query(text)
    print("Embedding dimension: ", len(embedding))
    print(embedding)


def embeddings_docs_demo():
    embeds = embeddings.embed_documents(
        [
            "This is first sample text to be embedded",
            "This is second sample text to be embedded",
        ]
    )
    print("Embeddings count: ", len(embeds))
    for i, embed in enumerate(embeds):
        print(f"Embed no #{i}")
        print(embed)


if __name__ == "__main__":
    # embeddings_demo()
    embeddings_docs_demo()

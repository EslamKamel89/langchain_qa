import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyMuPDFLoader,
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_core.documents import Document

load_dotenv()


def load_text_file():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(
            b"Hello this is a sample text file that is used for demonstration"
        )
        temp_file_path = temp_file.name
    try:
        loader = TextLoader(temp_file_path)
        documents = loader.load()
        for doc in documents:
            print("----------------------")
            print("Page Content: ")
            print(doc.page_content)
            print("Meta Data:")
            print(doc.metadata)
    finally:
        os.remove(temp_file_path)


def web_loader():
    loader = WebBaseLoader(
        "https://en.wikipedia.org/wiki/Web_scraping",
        bs_kwargs={"parse_only": None},
    )
    documents = loader.load()
    for doc in documents:
        print("-----------------------------")
        print("Document content: ")
        print(doc.page_content)
        print("Meta data: ")
        print(doc.metadata)


def lazy_loading():
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(5):
            path = Path(tmpdir) / f"doc_{i}.txt"
            path.write_text(f"this is document {i}. It contains sample content")
        loader = DirectoryLoader(tmpdir, glob="*.txt", loader_cls=TextLoader)
        documents = loader.lazy_load()
        for doc in documents:
            print("-------------------------------")
            print("Document Content: ")
            print(doc.page_content)
            print("Document Metadata: ")
            print(doc.metadata)


def doc_structure():
    document = Document(
        page_content="This is a sample content",
        metadata={
            "source": "sample_sources.txt",
            "author": "Eslam kamel",
        },
    )
    print("Document Content: ")
    print(document.page_content)
    print("Metadata: ")
    print(document.metadata)


def pdf_loader(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    for doc in documents:
        print("document content: ")
        print(doc.page_content)
        print("metadata:")
        print(doc.metadata)


if __name__ == "__main__":
    # load_text_file()
    # web_loader()
    # lazy_loading()
    # doc_structure()
    pdf_loader("./docs/cv.pdf")

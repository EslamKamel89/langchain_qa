from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    Language,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

load_dotenv()

SAMPLE_TEXT = """# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.

## Types of Machine Learning

### Supervised Learning
Supervised learning uses labeled data to train models. The algorithm learns to map inputs to outputs based on example input-output pairs.

Common algorithms include:
- Linear Regression
- Decision Trees
- Neural Networks

### Unsupervised Learning
Unsupervised learning finds hidden patterns in unlabeled data. The algorithm discovers structure without predefined labels.

Common algorithms include:
- K-Means Clustering
- Principal Component Analysis
- Autoencoders

## Applications

Machine learning is used in many fields:
1. Image recognition
2. Natural language processing
3. Recommendation systems
4. Fraud detection
5. Autonomous vehicles
""".strip()

SAMPLE_CODE = '''
def quicksort(arr):
    """
    Quicksort implementation in Python.
    Time complexity: O(n log n) average, O(n²) worst case.
    """
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quicksort(left) + middle + quicksort(right)


def binary_search(arr, target):
    """
    Binary search implementation.
    Requires sorted array.
    Time complexity: O(log n)
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
'''


def markdown_splitter():
    headers_to_consider = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
    splitter = MarkdownHeaderTextSplitter(headers_to_consider)
    for chunk in splitter.split_text(SAMPLE_TEXT):
        print("-" * 50 + "|" * 10 + "-" * 50)
        print("metadata: ", chunk.metadata)
        print("content: ", chunk)


def code_splitter():
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.PYTHON, chunk_size=200, chunk_overlap=20
    )
    for chunk in splitter.split_text(SAMPLE_CODE):
        print("-" * 50 + "|" * 10 + "-" * 50)
        print(chunk)


def document_splitter():
    loader = PyPDFLoader("./docs/cv.pdf")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    for chunk in chunks:
        print("-" * 50 + "|" * 10 + "-" * 50)
        print(chunk.page_content)


if __name__ == "__main__":
    # markdown_splitter()
    # code_splitter()
    document_splitter()

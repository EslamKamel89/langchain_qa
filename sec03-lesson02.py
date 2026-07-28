from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    Language,
    MarkdownTextSplitter,
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


def recursive_splitter():
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ".", ",", ""],
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_text(SAMPLE_TEXT)
    for chunk in chunks:
        print("-------------------------------")
        print(chunk)


def chunk_size_comparison():
    sizes = [200, 500, 1000]
    print("== Chunk Size Comparison ==")
    for size in sizes:
        print("\n\n")
        print(f"====================={size}=====================")
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ".", ",", ""],
            chunk_size=size,
            chunk_overlap=size // 5,
        )
        chunks = splitter.split_text(SAMPLE_TEXT)
        for chunk in chunks:
            print("    -------------------------")
            print("    ", chunk)


def overlap_importance():
    text = "The quick brown fox jumps over the lazy dog. " * 10
    no_overlap = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0)
    print("== No overlap splitter ==")
    chunks = no_overlap.split_text(text)
    for chunk in chunks:
        print("--------------")
        print(chunk)
    print("< No Overlap >")
    print(chunks[0][-20:])
    print(chunks[1][:20])

    print("\n\n")
    print("== overlap splitter ==")
    overlap = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=20)
    chunks = overlap.split_text(text)
    for chunk in chunks:
        print("--------------")
        print(chunk)
    print("< Overlap >")
    print(chunks[0][-20:])
    print(chunks[1][:20])


if __name__ == "__main__":
    # recursive_splitter()
    # chunk_size_comparison()
    overlap_importance()

import os
from enum import Enum
from typing import cast

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langsmith import Client, traceable
from pydantic import BaseModel, Field

load_dotenv()

model = init_chat_model("gpt-4o-mini", temperature=0)


def demo_basic_chain():
    prompt = ChatPromptTemplate.from_template(
        "Summarize the following test in one sentence: {text}"
    )
    parser = StrOutputParser()
    chain = prompt | model | parser
    res = chain.invoke(
        {
            "text": "LangChain is an open-source framework used to build applications with Large Language Models (LLMs). It serves as a bridge between a model and external data or software tools, allowing developers to create complex workflows—like chatbots that access private databases or AI agents that execute actions via APIs."
        }
    )
    print(res)


def demo_parallel_chain():
    summarize_prompt = ChatPromptTemplate.from_template(
        "Summarize in two sentences: {text}"
    )
    sentiment_prompt = ChatPromptTemplate.from_template(
        "what's the sentiment of the following text? {text}\n return as comma separated list"
    )
    keywords_prompt = ChatPromptTemplate.from_template(
        "Extract 5 keywords in the following text: {text}"
    )
    parser = StrOutputParser()
    parallel_chain = RunnableParallel(
        summary=summarize_prompt | model | parser,
        keywords=keywords_prompt | model | parser,
        sentiment=sentiment_prompt | model | parser,
    )
    text = """
    Models: LangChain provides a standardized, unified interface. This allows developers to seamlessly swap between different AI providers (e.g., OpenAI, Google Gemini, Anthropic) without rewriting large sections of code.Chains: Instead of just sending a single prompt and getting an answer, LangChain lets developers link multiple steps together. For example, a chain can automatically retrieve a document, summarize it, and then email it to a client.Prompt Templates: Instead of hard-coding prompts, developers can use templates that automatically plug in dynamic data (like a user's city, a name, or specific inputs).Retrieval-Augmented Generation (RAG): Standard LLMs can only use the data they were trained on. LangChain allows LLMs to access external data (like PDFs, websites, or company databases) by splitting them into chunks, embedding them, and searching them to fetch precise facts.
    """
    results = parallel_chain.invoke({"text": text})
    print(results)
    print("summary: ", results["summary"])
    print("keywords: ", results["keywords"])
    print("sentiment: ", results["sentiment"])


def fake_retriever(*args, **kwargs):
    return "Langchain is created by Harrison Chase in 2022"


def demo_passthrough_chain():
    prompt = ChatPromptTemplate.from_template("""
    Original question: {question}                                          
    Context: {context}
    Answer the question based on the context.
    """)
    chain = (
        RunnableParallel(
            context=RunnableLambda(fake_retriever),
            question=RunnablePassthrough(),
        )
        | RunnableLambda(
            lambda x: {"context": x["context"], "question": x["question"]["question"]}  # type: ignore
        )
        | prompt
        | model
        | StrOutputParser()
    )
    res = chain.invoke({"question": "Who created langchain"})
    print(res)


if __name__ == "__main__":
    # demo_basic_chain()
    # demo_parallel_chain()
    demo_passthrough_chain()

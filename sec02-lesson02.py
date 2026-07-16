import os
from enum import Enum
from typing import cast

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langsmith import Client, traceable
from pydantic import BaseModel, Field

load_dotenv()
model = init_chat_model("gpt-4o-mini", temperature=0.3)


def demo_chain_branching():
    code_prompt = ChatPromptTemplate.from_template(
        "You are a coding expert. Help with: {input}"
    )
    general_prompt = ChatPromptTemplate.from_template(
        "you are a helpful assistant help with: {input}"
    )
    classifier_prompt = ChatPromptTemplate.from_template(
        "Classify this as code or general: {input}\n return only the classification"
    )
    parser = StrOutputParser()
    classifier = classifier_prompt | model | parser

    def is_code_question(input_dict: dict):
        res = classifier.invoke(input_dict)
        print(f"Classifier: {res}")
        return res.lower() == "code"

    branch = RunnableBranch(
        (is_code_question, code_prompt | model | parser),
        general_prompt | model | parser,
    )
    questions = [
        "how to write for loops in python",
        "what's the weather today in egypt",
    ]
    for question in questions:
        print("-----------------------------------")
        print(f"Question: {question}")
        res = branch.invoke({"input": question})
        print(res)
        print("\n")


def demo_debugging():
    prompt = ChatPromptTemplate.from_template("Say hello to {name}")
    chain = prompt | model | StrOutputParser()
    # method 1:
    print(f"Chain input schema: {chain.input_schema.model_json_schema()}")
    print(f"Chain output schema: {chain.output_schema.model_json_schema()}")

    # method 2:
    result = chain.with_config(run_name="greeting_chain").invoke({"name": "Eslam"})
    print(f"Greeting result: {result}")

    # method 3
    def log_step(x, step_name=""):
        print(f"[{step_name}] {type(x).__name__}:{str(x)}")
        return x

    chain = (
        prompt
        | RunnableLambda(lambda x: log_step(x, "after_prompt"))
        | model
        | RunnableLambda(lambda x: log_step(x, "after_model"))
        | StrOutputParser()
    )
    res = chain.invoke({"name": "Eslam"})
    print(res)


if __name__ == "__main__":
    # demo_chain_branching()
    demo_debugging()

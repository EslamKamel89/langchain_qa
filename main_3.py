from typing import cast

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import (
    JsonOutputParser,
    PydanticOutputParser,
    StrOutputParser,
)
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

load_dotenv()

get_model = lambda: init_chat_model(model="gpt-4o-mini")


def demo_str_output_parser():
    prompt = ChatPromptTemplate.from_template("Write a short poem about {topic}")
    model = get_model()
    parser = StrOutputParser()
    chain = prompt | model | parser
    res = chain.invoke(
        {
            "topic": "You are not your thoughts you the one who is observing this thoughts"
        }
    )
    print(res)


def demo_json_output_parser():
    prompt = ChatPromptTemplate.from_template(
        "Return a json object with name and age for {description}"
    )
    model = get_model()
    parser = JsonOutputParser()
    chain = prompt | model | parser
    res = chain.invoke({"description": "My name is eslam and i am 37 years old"})
    print(res, type(res))
    print(res["name"])
    print(res["age"])


class Person(BaseModel):
    name: str = Field(..., description="The person's name")
    age: int = Field(..., description="The person's age")


def demo_pydantic_output_parser():
    parser = PydanticOutputParser(pydantic_object=Person)
    prompt = ChatPromptTemplate.from_template(
        "Return a json object with name and age for {description}"
    ).partial(format_instructions=parser.get_format_instructions())
    print(prompt.format_messages(description="My name is eslam and i am 37 years old"))
    model = get_model()
    chain = prompt | model | parser
    res = cast(
        Person, chain.invoke({"description": "My name is eslam and i am 37 years old"})
    )
    print(res)
    print(res.name)
    print(res.age)


class MovieReview(BaseModel):
    title: str = Field(..., description="The title of the movie")
    review: str = Field(..., description="A brief review of the movie")
    rating: int = Field(..., description="The rating of the movie out of 10")


def demo_movie_review():
    model = get_model().with_structured_output(MovieReview)
    prompt = ChatPromptTemplate.from_template("{description}")
    chain = prompt | model
    res = chain.invoke({"description": "Inception is an amazing thriller, 9/10"})
    res = cast(MovieReview, res)
    print(res.title)
    print(res.review)
    print(res.rating)


if __name__ == "__main__":
    # demo_str_output_parser()
    # demo_json_output_parser()
    # demo_pydantic_output_parser()
    demo_movie_review()

from typing import List, Optional, cast

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import (
    JsonOutputParser,
    PydanticOutputParser,
    StrOutputParser,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

model = init_chat_model("gpt-4o-mini", temperature=0.5)


def demo_str_parser():
    prompt = ChatPromptTemplate.from_template(
        "Give me a one-word answer: What color is the sky"
    )
    parser = StrOutputParser()
    chain = prompt | model | parser
    res = chain.invoke({})
    print(res)


def demo_json_parser():
    prompt = ChatPromptTemplate.from_template(
        "Return a JSON object with keys 'city' and 'country' for: {place}\n"
        "Return ONLY valid JSON, no explanation."
    )
    parser = JsonOutputParser()
    chain = prompt | model | parser
    result = chain.invoke({"place": "The Eiffel Tower"})
    print(type(result), result["city"], result["country"])


def demo_pydantic_parser():
    class Recipe(BaseModel):
        name: str = Field(description="Name of the recipe")
        ingredients: List[str] = Field(description="List of ingredients")
        prep_time_minutes: int = Field(description="Preparation time in minutes")
        difficulty: str = Field(description="easy, medium, or hard")

    parser = PydanticOutputParser(pydantic_object=Recipe)
    prompt = ChatPromptTemplate.from_template(
        "Create a simple recipe for: {dish}\n\n{format_instructions}"
    ).partial(format_instructions=parser.get_format_instructions())
    chain = prompt | model | parser
    result = chain.invoke({"dish": "scrambled eggs"})
    print(f"Recipe: {result.name}")
    print(f"Ingredients: {result.ingredients}")
    print(f"Prep time: {result.prep_time_minutes} mins")
    print(f"Difficulty: {result.difficulty}")


def demo_structured_output():
    class TextExtraction(BaseModel):
        task: str = Field(description="The main task to do")
        priority: str = Field(description="high, medium, or low")
        deadline: Optional[str] = Field(description="Deadline if mentioned")
        assignee: Optional[str] = Field(description="Person assigned if mentioned")

    structured_model = model.with_structured_output(TextExtraction)
    prompt = ChatPromptTemplate.from_template("Extract task information from: {text}")
    chain = prompt | structured_model
    texts = [
        "John needs to finish the report by Friday - it's urgent",
        "We should update the docs sometime next week",
        "Critical: Fix the login bug ASAP",
    ]

    print("Task Extractions:")
    for text in texts:
        result = cast(TextExtraction, chain.invoke({"text": text}))
        print(f"\nInput: {text}")
        print(result)
        print(f"  Task: {result.task}")
        print(f"  Priority: {result.priority}")
        print(f"  Deadline: {result.deadline}")
        print(f"  Assignee: {result.assignee}")


def demo_complex_schema():
    class Address(BaseModel):
        street: str
        city: str
        country: str

    class Company(BaseModel):
        name: str
        industry: str
        employee_count: int
        headquarters: Address
        products: list[str]

    structured_model = model.with_structured_output(Company)
    prompt = ChatPromptTemplate.from_template(
        "Extract company information from: {text}"
    )
    chain = prompt | structured_model
    res = chain.invoke(
        {
            "text": "Apple Inc. is a tech company with 160,000 employees based in "
            "Cupertino, California, USA. They make iPhones, MacBooks, and iPads."
        }
    )
    res = cast(Company, res)
    print(f"Company: {res.name}")
    print(f"Industry: {res.industry}")
    print(f"Employees: {res.employee_count}")
    print(f"HQ: {res.headquarters.city}, {res.headquarters.country}")
    print(f"Products: {res.products}")


def exercise_structured_extraction():
    class Movie(BaseModel):
        title: str = Field(description="Movie title")
        year: int = Field(description="Year released")
        director: str = Field(description="Director name")
        actors: List[str] = Field(description="Main actors")
        genre: str = Field(description="Primary genre")
        rating: int = Field(description="Rating from 1-10", ge=1, le=10)

    structured_model = model.with_structured_output(Movie)
    prompt = ChatPromptTemplate.from_template(
        "Extract movie information from this review:\n\n{review}"
    )

    chain = prompt | structured_model
    result = chain.invoke(
        {
            "review": "The Dark Knight (2008) directed by Christopher Nolan is an "
            "absolute masterpiece. Christian Bale and Heath Ledger deliver "
            "incredible performances in this action thriller. 10/10!"
        }
    )
    result = cast(Movie, result)
    print(f"Title: {result.title}")
    print(f"Year: {result.year}")
    print(f"Director: {result.director}")
    print(f"Actors: {result.actors}")
    print(f"Genre: {result.genre}")
    print(f"Rating: {result.rating}/10")


if __name__ == "__main__":
    # demo_str_parser()
    # demo_json_parser()
    # demo_pydantic_parser()
    # demo_structured_output()
    # demo_complex_schema()
    exercise_structured_extraction()

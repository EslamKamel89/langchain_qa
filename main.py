import asyncio

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()


async def demo_basic_chain():
    """Demonstrates a basic chain using LCEL and runnables."""
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistance. Answer in one sentence: {question}"
    )
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    parser = StrOutputParser()
    chain = prompt | model | parser

    result = chain.invoke({"question": "What's langchain"})
    print("Result: ", result)
    return chain


async def demo_batch_execution():
    """Demonstrated batch execution for multiple inputs"""
    prompt = ChatPromptTemplate.from_template(
        "Translate to french: {text}. respond only with the translated text"
    )
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
    parser = StrOutputParser()
    chain = prompt | model | parser
    inputs = [
        {"text": text}
        for text in [
            "How are you?",
            "What's your name?",
            "Where is the nearest restaurant?",
        ]
    ]
    results = chain.batch(inputs)
    for input, result in zip(inputs, results):
        print(f"Input: {input['text']} => Output: {result}")


async def demo_streaming():
    """Demonstrates steaming for real time output"""
    prompt = ChatPromptTemplate.from_template("Tell me a story about: {topic}")
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    parser = StrOutputParser()
    chain = prompt | model | parser
    print("Streaming output: ")
    for chunk in chain.stream(
        {"topic": "Steer attention + Do Well + Be Kind = Shape Your life"}
    ):
        print(chunk, end="", flush=True)
        # print('')


async def schema_inspection():
    """Demonstrates input/output schema inspection"""
    prompt = ChatPromptTemplate.from_template("Summarize the following text: {text}")
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    parser = StrOutputParser()

    chain = prompt | model | parser

    input_schema = chain.input_schema.model_json_schema()
    output_schema = chain.output_schema.model_json_schema()

    print(f"Input Schema: {input_schema}")
    print(f"Output Schema: {output_schema}")


async def exercise_first_chain():
    prompt = ChatPromptTemplate.from_template(
        "Generate a marketing tagline for this product {name} and this audience {audience}"
    )
    model = ChatOpenAI()
    parser = StrOutputParser()
    chain = prompt | model | parser
    res = chain.invoke(
        {
            "name": "AI Course",
            "audience": "developers",
        }
    )
    print(res)


from langchain.chat_models import init_chat_model


async def new_way_to_initialize_model():
    prompt = ChatPromptTemplate.from_template(
        "Generate a marketing tagline for this product {name} and this audience {audience}"
    )
    model = init_chat_model(
        "gpt-4o-mini",
        model_provider="openai",
        temperature=0.7,
        max_tokens=1500,
    )
    parser = StrOutputParser()
    chain = prompt | model | parser
    res = chain.invoke(
        {
            "name": "AI Course",
            "audience": "developers",
        }
    )
    print(res)


async def configure_models():
    prompt = ChatPromptTemplate.from_template(
        "explain the biggest insight in `your are not minds` and `you are the observer of your thoughts` wisdoms"
    )
    model = ChatOpenAI(
        name="gpt-4o-mini",
        temperature=0.7,
        max_completion_tokens=1500,
        timeout=30,
        max_retries=3,
    )
    parser = StrOutputParser()
    chain = prompt | model | parser
    for chunk in chain.stream({}):
        print(chunk, end="", flush=True)


def get_model():
    return init_chat_model(
        model="gpt-4o-mini",
        model_provider="openai",
        temperature=0.7,
        max_tokens=1500,
        streaming=True,
    )


from langchain.messages import HumanMessage, SystemMessage


async def system_and_human_messages():
    messages = [
        SystemMessage("You are a helpful assistant"),
        HumanMessage("Respond in one word only, what is the capital of France"),
    ]
    model = get_model()
    res = model.invoke(messages)
    print(res.content)


from langchain_core.messages import AIMessage
from langchain_core.prompts import MessagesPlaceholder


async def system_and_human_messages2():
    messages = [
        SystemMessage("You are a helpful assistant"),
        MessagesPlaceholder("messages"),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)

    chain = prompt | get_model() | StrOutputParser()
    conversation = []
    conversation.append(
        HumanMessage("Respond in one word only, what is the capital of France")
    )
    answer = chain.invoke({"messages": conversation})
    print(answer)
    conversation.append(AIMessage(answer))
    conversation.append(
        HumanMessage("What country did I ask you to find the capital of?")
    )
    answer = chain.invoke({"messages": conversation})
    print(answer)


if __name__ == "__main__":
    # asyncio.run(demo_basic_chain())
    # asyncio.run(demo_batch_execution())
    # asyncio.run(demo_streaming())
    # asyncio.run(schema_inspection())
    # asyncio.run(exercise_first_chain())
    # asyncio.run(new_way_to_initialize_model())
    # asyncio.run(configure_models())
    # asyncio.run(system_and_human_messages())
    asyncio.run(system_and_human_messages2())

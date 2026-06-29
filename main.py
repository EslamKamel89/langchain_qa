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


def exercise_multi_model():
    def get_response(question: str, models: list[str]) -> dict[str, str]:
        prompt = ChatPromptTemplate.from_template("{message}")
        parser = StrOutputParser()
        responses = {}
        for model_name in models:
            model = init_chat_model(
                model=model_name,
                temperature=0.7,
                streaming=False,
            )
            chain = prompt | model | parser
            res = chain.invoke({"message": question})
            responses[model_name] = res
        return responses

    results = get_response("What Is AI?", ["gpt-4o-mini", "gpt-4o"])
    for model, answer in results.items():
        print(f"Response from {model}: {answer}")


def hands_on_prompt_templating():
    prompt = ChatPromptTemplate.from_template("Tell me {adjective} job about {topic}")
    # chain = prompt | get_model() | StrOutputParser()
    # res = chain.invoke({"adjective": "funny", "topic": "ai replacing programmers"})
    # print(res)
    messages = prompt.format_messages(adjective="funny", topic="programming")
    print(messages)


def hands_on_multi_message_templates():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a translator that translates from {input_language} to {output_language}",
            ),
            ("human", "Translate the following : {text}"),
        ]
    )
    messages = prompt.format_messages(
        input_language="english", output_language="arabic", text="I love programming"
    )
    print(messages)
    chain = get_model() | StrOutputParser()
    res = chain.invoke(messages)
    print(res)


from langchain_core.prompts import FewShotChatMessagePromptTemplate


def hands_on_multi_message_templates2():
    examples = [
        {"input": "happy", "output": "sad"},
        {"input": "tall", "output": "short"},
    ]
    example_prompt = ChatPromptTemplate.from_messages(
        [("human", "{input}"), ("ai", "{output}")]
    )
    fewshot_prompt = FewShotChatMessagePromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
    )
    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Give the opposite of each word."),
            fewshot_prompt,
            ("human", "{input}"),
        ]
    )
    print(final_prompt.format_messages(input="white"))
    chain = final_prompt | get_model() | StrOutputParser()
    res = chain.invoke({"input": "white"})
    print(res)


if __name__ == "__main__":
    # asyncio.run(demo_basic_chain())
    # asyncio.run(demo_batch_execution())
    # asyncio.run(demo_streaming())
    # asyncio.run(schema_inspection())
    # asyncio.run(exercise_first_chain())
    # asyncio.run(new_way_to_initialize_model())
    # asyncio.run(configure_models())
    # asyncio.run(system_and_human_messages())
    # asyncio.run(system_and_human_messages2())
    # exercise_multi_model()
    # hands_on_prompt_templating()
    # hands_on_multi_message_templates()
    hands_on_multi_message_templates2()

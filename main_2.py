from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

load_dotenv()


def demo_basic_template():
    simple_prompt = ChatPromptTemplate.from_template("Translate '{text}' to {language}")
    print("Simple Prompt: ")
    print(simple_prompt.format_messages(text="Hello world!!", language="french"))
    multiple_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a translator, be concise"),
            ("human", "Translate `{text}` to {language}"),
        ]
    )
    print("Multi Part Prompt: ")
    messages = multiple_prompt.format_messages(
        text="I love programming", language="french"
    )
    for message in messages:
        print(f"{type(message).__name__}: {message.content}")


def demo_message_types():
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    messages = [
        SystemMessage("You are a math tutor, be brief"),
        HumanMessage("What's 5*5 ?"),
        AIMessage("25"),
        HumanMessage("And if i add 10"),
    ]
    res = model.invoke(messages)
    print(res.content)


def demo_messages_placeholder():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant"),
            MessagesPlaceholder("history"),
            ("human", "{question}"),
        ]
    )
    history = [
        HumanMessage("My name is eslam"),
        AIMessage("Nice to meet you eslam"),
    ]
    model = init_chat_model(
        model="gpt-4o-mini",
        temperature=0.7,
    )
    chain = prompt | model | StrOutputParser()
    res = chain.invoke({"history": history, "question": "What is my name"})
    print(res)


if __name__ == "__main__":
    # demo_basic_template()
    # demo_message_types()
    demo_messages_placeholder()

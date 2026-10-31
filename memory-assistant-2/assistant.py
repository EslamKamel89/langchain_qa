import os
from typing import Annotated, NotRequired, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage
from langchain_core.messages import AIMessage, AnyMessage, RemoveMessage, trim_messages
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph, add_messages

load_dotenv()

WINDOW_SIZE = 4  # Four messages, roughly two exchanges
MAX_CONTEXT_TOKENS = 1200
SUMMARY_THRESHOLD = 10
KEEP_RECENT = 4

model = init_chat_model(
    model=os.getenv("OPENAI_MODEL"),
    model_provider="openai",
    temperature=0.0,
)

chat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a developer assistant.

Be accurate, concise, and technical.

Known information from earlier conversation:
{summary}

Use this information and the recent conversation
to maintain continuity.

If information is missing, say so.
Do not invent user preferences.""",
        ),
        MessagesPlaceholder("history"),
    ]
)
chat_chain = chat_prompt | model | StrOutputParser()

summary_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You maintain a factual conversation summary.
    
    Update the existing summary using the new messages.
    
    Preserve:
    - User identity and preferences
    - Technical background
    - Active projects and goals
    - Important decisions and constraints
    
    Do not invent facts.
    When facts conflict, prefer the latest explicit correction.
    Keep the summary concise.
    
    Existing summary:
    {summary}
    
    New conversation:
    {conversation}""",
        )
    ]
)

summary_chain = summary_prompt | model | StrOutputParser()


class AssistantState(TypedDict):
    summary: NotRequired[str]
    messages: Annotated[list[AnyMessage], add_messages]


def prepare_context(messages: list[AnyMessage]) -> list[AnyMessage]:
    return trim_messages(
        messages,
        max_tokens=MAX_CONTEXT_TOKENS,
        token_counter=model,  # type: ignore
        strategy="last",
        include_system=False,
        allow_partial=False,
        start_on="human",
    )  # type: ignore


def chat_node(state: AssistantState) -> dict:
    context = prepare_context(state["messages"])

    response = chat_chain.invoke(
        {
            "summary": state.get("summary", ""),
            "history": context,
        }
    )

    return {"messages": [AIMessage(content=response)]}


def summarize_node(state: AssistantState) -> dict:
    messages = state["messages"]
    if len(messages) <= SUMMARY_THRESHOLD:
        return {}

    old_messages = messages[:-KEEP_RECENT]
    conversation = "\n".join("" for message in old_messages)
    updated_summary = summary_chain.invoke(
        {
            "summary": state.get("summary"),
            "conversation": conversation,
        }
    )
    return {
        "summary": updated_summary,
        "messages": [RemoveMessage(id=message.id) for message in old_messages],  # type: ignore
    }


def build_graph(checkpointer: SqliteSaver):
    builder = StateGraph(state_schema=AssistantState)
    builder.add_node("chat", chat_node)
    builder.add_node("summarize", summarize_node)
    builder.add_edge(START, "chat")
    builder.add_edge("chat", "summarize")
    builder.add_edge("summarize", END)
    return builder.compile(checkpointer=checkpointer)

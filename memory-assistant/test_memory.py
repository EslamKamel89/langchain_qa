from assistant import build_graph
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver


def test_thread_isolation():
    graph = build_graph(InMemorySaver())

    config_a = {"configurable": {"thread_id": "test-a"}}

    config_b = {"configurable": {"thread_id": "test-b"}}

    graph.invoke(
        {"messages": [HumanMessage(content="I prefer Python.")]},
        config_a,
    )

    graph.invoke(
        {"messages": [HumanMessage(content="I prefer JavaScript.")]},
        config_b,
    )

    state_a = graph.get_state(config_a).values
    state_b = graph.get_state(config_b).values

    assert "Python" in state_a["messages"][0].content
    assert "JavaScript" in state_b["messages"][0].content

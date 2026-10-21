import argparse

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver

from .assistant import build_graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thread", required=True, help="Conversation identifier")
    args = parser.parse_args()
    config: RunnableConfig = {"configurable": {"thread_id": args.thread}}
    with SqliteSaver.from_conn_string(
        "./memory-assistant/checkpoints.sqlite"
    ) as checkpointer:  # type: ignore
        checkpointer.setup()
        graph = build_graph(checkpointer)
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                break
            if user_input == "/memory":
                snapshot = graph.get_state(config)
                state = snapshot.values

                print("\n--- SUMMARY ---")
                print(state.get("summary", ""))

                print("\n--- RECENT MESSAGES ---")
                for message in state.get("messages", []):
                    print(f"{message.type}: {message.content}")
                continue

            if not user_input:
                continue
            result = graph.invoke(
                {"messages": [HumanMessage(content=user_input)]}, config=config
            )
            print("AI:", result["messages"][-1].content)


if __name__ == "__main__":
    main()

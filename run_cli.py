r"""Simple CLI to interact with the Agent in `agent/graph.py`.

Usage (Windows cmd):
    cd /d F:\langgraph_excel_chatbot
    python run_cli.py           # non-streaming
    python run_cli.py --stream  # streaming mode (yields chunks)

Type 'exit' or 'quit' to end the session.
"""

import argparse
import sys
import uuid


def main(stream: bool = False):
    # Import here to avoid package load at module import time
    try:
        from agent import graph
    except Exception as e:
        print("Failed to import agent.graph:\n", e)
        sys.exit(1)

    agent = graph.agent

    print(
        "Hi I am expert data analyst AI agent I am here to assist you in analysis of your three tables brand proposals,brand payments,brand contacts"
    )
    # Visualization commands disabled. Use natural language to request data and the agent will return tables via query_db.

    # create a session id for this CLI run so conversation persists
    session_id = str(uuid.uuid4())

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not user_input:
            continue
        # Removed special-case typed exit/quit so the loop runs until manually
        # terminated (Ctrl+C).

        # Visualizations disabled: only use natural language to request data (query_db)

        if stream:
            try:
                for chunk in agent.stream(user_input, session_id=session_id):
                    print(chunk, end="", flush=True)
                print("\n")
            except Exception as e:
                print("\nError during streaming:", e)
        else:
            try:
                resp = agent.invoke(user_input, session_id=session_id)
                print("\nAgent:", resp)
            except Exception as e:
                print("\nError invoking agent:", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", action="store_true", help="Stream responses")
    args = parser.parse_args()
    main(stream=args.stream)

# LangGraph Excel Chatbot (Python)

Minimal NL -> SQL agent intended to be used with LangGraph or as a standalone HTTP service.

What this provides

- `langgraph_agent.py` — async function `handle_query(user_query: str)` that:
  - inspects the `public` schema of your Postgres (Supabase) database
  - calls OpenAI to translate natural language to a single Postgres `SELECT` SQL statement
  - executes the SQL and returns rows
- `app.py` — a small FastAPI wrapper for testing the agent via HTTP

Requirements

- Python 3.10+
- A Postgres connection string (Supabase DATABASE_URL)
- An OpenAI API key

Setup (Windows cmd)

1. Create a virtual environment and activate it

```
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies

```
pip install -r requirements.txt
```

3. Create a `.env` file (copy `.env.example`) and fill in your credentials:

```
OPENAI_API_KEY=sk-...
DATABASE_URL=postgresql://user:password@host:5432/database
PORT=8000
```

4. Run the FastAPI app for local testing

```
uvicorn app:app --host 0.0.0.0 --port 8000
```

HTTP endpoints

- GET / -> basic health message
- POST /query -> body: { "query": "natural language request" }
  - Returns: { sql, rows, rowCount } or an error

Example curl (from another shell):

```
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d "{\"query\": \"list top 5 customers by revenue\"}"
```

Using from Python / LangGraph

The agent exposes an async function `handle_query` which you can call directly from Python or integrate into a LangGraph flow. Example:

```py
import asyncio
from langgraph_agent import handle_query

async def main():
    res = await handle_query('list top 5 customers by revenue')
    print(res)

asyncio.run(main())
```

To integrate into LangGraph, wrap `handle_query` as a node or function step in your LangGraph flow. For example (pseudocode):

```py
# pseudo-code — adapt to your LangGraph setup
from langgraph import Flow, FunctionNode
from langgraph_agent import handle_query

async def nl_to_sql_node(input_text: str):
    return await handle_query(input_text)

flow = Flow()
flow.add_node(FunctionNode(nl_to_sql_node))
```

Safety and notes

- This starter enforces only `SELECT` queries for safety. If you need write access, add authentication and strong validation.
- The OpenAI prompt and model choice are intentionally minimal. Tune the prompt, temperature, or model to match your needs.

Next steps you might want

- Add caching of schema to reduce DB calls
- Improve prompt engineering to get safer/shorter SQL
- Add role-based access control for risky queries

Running `agent/graph.py` (import error note)

If you run `agent/graph.py` directly from the `agent/` folder you might see:

ModuleNotFoundError: No module named 'agent'

This happens because Python's import system finds top-level packages relative to where the interpreter is started. Recommended ways to run the graph without modifying sys.path:

1. From the project root, run as a module (preferred):

python -m agent.graph

1. From the project root set PYTHONPATH to the project root then run the file:

set PYTHONPATH=%CD%
python agent\graph.py

1. Keep using package-style imports — the repository uses relative imports inside `agent/` so running via `python -m agent.graph` is the most robust option.

If you prefer running the file directly from inside `agent/` (not recommended), change the imports in `agent/graph.py` to use relative imports (`from .tools import ...` etc.). The repository has been updated to use relative imports already.

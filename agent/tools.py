"""
Tools for the agent to use.
"""

import datetime
import os
import tempfile
import traceback
import uuid
from types import SimpleNamespace

import pandas as pd
from dotenv import load_dotenv
from langchain_core.tools import tool
from sqlalchemy import Engine, create_engine, text

# Load .env into environment so os.getenv can pick up values from your .env file
load_dotenv()

# Provide env with SUPABASE_URL read from the environment; prefer SUPABASE_URL then DATABASE_URL
_supabase_url = os.getenv("SUPABASE_URL") or os.getenv("DATABASE_URL")
if not _supabase_url:
    # sensible fallback for local dev
    _supabase_url = "postgresql://user:password@localhost:5432/db"

env = SimpleNamespace(SUPABASE_URL=_supabase_url)

from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command


class ServerSession:
    """A session for server-side state management and operations.

    In practice, this would be a separate service from where the agent is running and the agent would communicate with it using a REST API. In this simplified example, we use it to persist the db engine and data returned from the query_db tool.
    """

    def __init__(self):
        self.engine: Engine = None
        self.df: pd.DataFrame = None

        self.engine = self._get_engine()

    def _get_engine(self):
        if self.engine is None:
            # Configure SQLAlchemy for session pooling
            _engine = create_engine(
                env.SUPABASE_URL,
                pool_size=5,  # Smaller pool size since the pooler manages connections
                max_overflow=5,  # Fewer overflow connections needed
                pool_timeout=10,  # Shorter timeout for getting connections
                pool_recycle=1800,  # Recycle connections more frequently
                pool_pre_ping=True,  # Keep this to verify connections
                pool_use_lifo=True,  # Keep LIFO to reduce number of open connections
                connect_args={
                    "application_name": "onlyvans_agent",
                    "options": "-c statement_timeout=30000",
                    # Keepalives less important with transaction pooler but still good practice
                    "keepalives": 1,
                    "keepalives_idle": 60,
                    "keepalives_interval": 30,
                    "keepalives_count": 3,
                },
            )
            return _engine
        return self.engine


# Create a global instance of the ServerSession
session = ServerSession()


@tool
def query_db(query: str) -> str:
    """Query the database using Postgres SQL.

    Args:
        query: The SQL query to execute. Must be a valid postgres SQL string that can be executed directly.

    Returns:
        str: The query result as a markdown table.
    """
    try:
        # debug: write incoming query to a temp debug file to avoid repo output writes
        try:
            dbg_path = os.path.join(tempfile.gettempdir(), "query_debug.txt")
            with open(dbg_path, "a", encoding="utf-8") as f:
                f.write("--- QUERY START ---\n")
                f.write(query + "\n")
        except Exception:
            # ignore debug write failures to avoid crashing the tool
            pass

        # Use the global engine in the server session to connect to Supabase
        with session.engine.connect().execution_options(
            isolation_level="READ COMMITTED"
        ) as conn:
            result = conn.execute(text(query))

            columns = list(result.keys())
            rows = result.fetchall()
            df = pd.DataFrame(rows, columns=columns)

            # Store the DataFrame in the server session
            session.df = df

            # debug: record shape and a small head of the dataframe into temp dir
            try:
                dbg_path = os.path.join(tempfile.gettempdir(), "query_debug.txt")
                with open(dbg_path, "a", encoding="utf-8") as f:
                    f.write(f"Fetched rows: {len(df)}; columns: {columns}\n")
                    try:
                        f.write(df.head(20).to_csv(index=False))
                    except Exception:
                        f.write(repr(df.head(20)) + "\n")
                    f.write("\n--- QUERY END ---\n\n")
            except Exception:
                pass

            conn.close()  # Explicitly close the connection

        # Do not persist full CSVs to the repo 'output/' folder. Keep csv_path None
        csv_path = None

        # Return chunked markdown tables to avoid client-side rendering limits
        try:
            total_rows = len(df)
            header = f"Query returned {total_rows} rows."
            if csv_path:
                header += f" CSV saved to: {csv_path}"

            # Choose a conservative chunk size (rows per markdown table)
            chunk_size = 1000
            if total_rows == 0:
                return header + "\n\nNo rows returned."

            md_parts = [header, "\n"]
            for start in range(0, total_rows, chunk_size):
                end = min(start + chunk_size, total_rows)
                chunk = df.iloc[start:end]
                md_parts.append(f"-- rows {start + 1} to {end} --")
                try:
                    md_parts.append(chunk.to_markdown(index=False))
                except Exception:
                    # fallback to CSV snippet if markdown fails
                    md_parts.append(chunk.to_csv(index=False))

            md = "\n\n".join(md_parts)
            return md
        except Exception as e:
            # as a last resort return CSV text
            try:
                return df.to_csv(index=False)
            except Exception:
                return f"Error converting results to markdown or CSV: {e}"
    except Exception as e:
        return f"Error executing query: {str(e)}"


@tool
def generate_visualization(
    name: str,
    sql_query: str,
    plotly_code: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command | str:
    """Visualizations are disabled in this environment.

    This stub exists to ensure the agent can run safely without Plotly/Kaleido.
    If a user requests a visualization, return a clear message instructing them how to
    enable the feature (install dependencies and re-enable the tool).
    """
    return "Visualization support is disabled. To enable, install plotly and kaleido and re-enable the generate_visualization tool."

import os
import uuid
from typing import Annotated, Generator, List

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel
from sqlalchemy import text as sa_text

from .groq_client import ChatGROQ  # ✅ Custom Groq wrapper
from .prompts import prompts
from .tools import query_db, session


class ScoutState(BaseModel):
    """LangGraph state model that holds conversation messages and chart data."""

    messages: Annotated[List[BaseMessage], add_messages] = []
    chart_json: str = ""


class Agent:
    """
    LangGraph-based agent using Groq as the LLM backend.
    Supports querying data + visualization generation.
    """

    def __init__(
        self,
        name: str,
        tools: List = [query_db],
        model: str = "openai/gpt-oss-20b",  # ✅ Groq model (can be overridden by GROQ_MODEL env var)
        system_prompt: str = "You are a helpful assistant for Excel and data analysis.",
        temperature: float = 0.1,
    ):
        self.name = name
        self.tools = tools
        # Allow overriding the default model via the environment for quick switches
        self.model = os.getenv("GROQ_MODEL", model)
        self.system_prompt = system_prompt
        self.temperature = temperature

        # ✅ Initialize Groq LLM
        self.llm = ChatGROQ(model=self.model, temperature=self.temperature).bind_tools(
            self.tools
        )

        # ✅ Build LangGraph workflow
        self.runnable = self.build_graph()

    # -----------------------
    # Build Graph
    # -----------------------
    def build_graph(self):
        """Build the LangGraph computation graph."""

        def scout_node(state: ScoutState) -> ScoutState:
            """Run LLM on current conversation state."""
            # build messages list safely to avoid static type issues
            msgs = [SystemMessage(content=self.system_prompt)]
            msgs.extend(state.messages or [])
            response = self.llm.invoke(msgs)
            state.messages = state.messages + [response]
            return state

        def router(state: ScoutState) -> str:
            """Decide whether to call a tool or finish."""
            last_message = state.messages[-1]
            if not getattr(last_message, "tool_calls", None):
                return END
            return "tools"

        builder = StateGraph(ScoutState)
        builder.add_node("chatbot", scout_node)
        builder.add_node("tools", ToolNode(self.tools))
        builder.add_edge(START, "chatbot")
        builder.add_conditional_edges("chatbot", router, ["tools", END])
        builder.add_edge("tools", "chatbot")

        # Configure a namespaced checkpoint for the MemorySaver to avoid
        # runtime errors when no checkpoint config is provided.
        # By default we do not enable durable checkpointing for local runs
        # because the checkpointer implementation may require configurable keys
        # (thread_id, checkpoint_ns, checkpoint_id). Enable by setting
        # ENABLE_CHECKPOINTS=1 in the environment and providing CHECKPOINT_NS/ID.
        if os.getenv("ENABLE_CHECKPOINTS", "0") == "1":
            checkpoint_ns = os.getenv("CHECKPOINT_NS", "local")
            checkpoint_id = os.getenv("CHECKPOINT_ID", str(uuid.uuid4()))

            saver = MemorySaver()
            if hasattr(saver, "configure") and callable(getattr(saver, "configure")):
                try:
                    saver.configure(
                        checkpoint_ns=checkpoint_ns, checkpoint_id=checkpoint_id
                    )
                except Exception:
                    try:
                        saver.configure(
                            namespace=checkpoint_ns, checkpoint_id=checkpoint_id
                        )
                    except Exception:
                        pass
            else:
                try:
                    setattr(saver, "checkpoint_ns", checkpoint_ns)
                    setattr(saver, "checkpoint_id", checkpoint_id)
                except Exception:
                    pass

            return builder.compile(checkpointer=saver)

        # Default: no checkpointer for local/dev runs
        return builder.compile(checkpointer=None)

    # -----------------------
    # Debug: visualize graph
    # -----------------------
    def inspect_graph(self):
        """Visualize the agent graph (for debugging in Jupyter)."""
        try:
            from IPython.display import Image, display

            graph = self.build_graph()
            display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
        except Exception:
            # IPython not available in this environment; just return
            return

    # -----------------------
    # Runtime: invoke / stream
    # -----------------------
    def invoke(self, message: str, session_id: str | None = None, **kwargs) -> str:
        """Run one synchronous interaction and optionally persist conversation by session_id.

        If `session_id` is None a new UUID will be generated and returned in the assistant reply so callers can reuse it.
        """
        session_id = session_id or str(uuid.uuid4())

        # Ensure conversation table exists (best-effort)
        try:
            self._ensure_conversation_table()
        except Exception:
            pass

        # Load past messages for this session (if any) so the LLM has context
        try:
            history_messages = self._load_conversation_messages(session_id)
        except Exception:
            history_messages = []

        # Build message list: history + current human message
        user_msg = HumanMessage(content=message)
        messages = (history_messages or []) + [user_msg]

        # Persist the user message (best-effort)
        try:
            self._save_message(session_id, "user", message)
        except Exception:
            pass

        result = self.runnable.invoke(input={"messages": messages}, **kwargs)
        last_message = result["messages"][-1]

        # If the LLM requested a tool call, execute it directly when possible.
        tool_calls = getattr(last_message, "tool_calls", None)
        if tool_calls:
            # tool_calls may be a list; handle the first one for synchronous flow
            tc = tool_calls[0]
            try:
                tool_name = (
                    tc.get("name")
                    if isinstance(tc, dict)
                    else getattr(tc, "name", None)
                )
                args = (
                    tc.get("args")
                    if isinstance(tc, dict)
                    else getattr(tc, "args", None)
                )
            except Exception:
                tool_name = None
                args = None

                if tool_name:
                    exec_result = self._execute_tool_call(tool_name, args)
                    # persist tool result as assistant content
                    try:
                        self._save_message(session_id, "assistant", str(exec_result))
                    except Exception:
                        pass
                    return str(exec_result)

        # persist assistant message (strip Plan/analysis sections before saving)
        cleaned_assistant = None
        try:
            raw_assistant = getattr(last_message, "content", "") or ""
            cleaned_assistant = self._strip_plan_sections(raw_assistant)
            self._save_message(session_id, "assistant", cleaned_assistant)
        except Exception:
            # leave cleaned_assistant as None so we can safely fallback below
            cleaned_assistant = None

        # Return assistant content (session_id is already in use for persistence).
        return cleaned_assistant or (last_message.content or "")

    def _extract_sql_from_text(self, text: str) -> str | None:
        """Try to extract SQL from a block in the assistant text.

        Looks for ```sql ... ``` fenced blocks first, then generic ```...``` fences,
        then falls back to a simple regex for SELECT ... FROM patterns.
        """
        try:
            import re

            # fenced ```sql blocks
            m = re.search(r"```sql\s+(.*?)```", text, re.DOTALL | re.IGNORECASE)
            if m:
                return m.group(1).strip()

            # generic fenced code block
            m = re.search(r"```\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
            if m:
                candidate = m.group(1).strip()
                # a quick heuristic: must contain SELECT and FROM
                if re.search(r"\bSELECT\b", candidate, re.IGNORECASE) and re.search(
                    r"\bFROM\b", candidate, re.IGNORECASE
                ):
                    return candidate

            # fallback: find first SELECT ... FROM ...; limit to reasonable length
            m = re.search(
                r"(SELECT[\s\S]{10,2000}?FROM[\s\S]{0,2000}?)(?:;|$)",
                text,
                re.IGNORECASE,
            )
            if m:
                return m.group(1).strip()
        except Exception:
            return None
        return None

    def _strip_plan_sections(self, text: str) -> str:
        """Remove explicit planning/analysis sections from assistant text.

        This should be resilient to missing sections and not raise.
        It strips headings like "Plan:", "Analysis:", and any following
        bullet/numbered lists or paragraphs that look like a planning block.
        """
        if not text:
            return ""
        try:
            import re

            s = str(text)

            # Remove common Plan/Analysis sections and everything after them up
            # to a reasonable delimiter (two newlines or end of text).
            # This pattern looks for a heading like 'Plan:' or 'Analysis:'
            # and captures everything following until a blank line-double newline
            # or end-of-string. We keep the content before the plan section.
            m = re.search(r"(?si)(.*?)(?:\n\s*(?:Plan|Analysis)\s*:\s*)([\s\S]*)$", s)
            if m:
                # return only the portion before the plan/analysis heading
                before = m.group(1).strip()
                if before:
                    return before

            # Fallback: remove lines that look like step-by-step plan markers
            # e.g., numbered lists or lines starting with '-' or '*' that are
            # contiguous and appear near the start of the assistant response.
            lines = s.splitlines()
            cleaned_lines = []
            in_plan_block = False
            plan_indicators = ("plan:", "analysis:")
            for ln in lines:
                stripped = ln.strip()
                low = stripped.lower()
                if any(low.startswith(pi) for pi in plan_indicators):
                    # stop collecting further lines when we hit a plan header
                    in_plan_block = True
                    break
                cleaned_lines.append(ln)

            cleaned = "\n".join(cleaned_lines).strip()
            # If nothing meaningful remains, return original text to avoid
            # returning an empty assistant reply.
            return cleaned if cleaned else s
        except Exception:
            return str(text)

    def stream(self, message: str, **kwargs) -> Generator[str, None, None]:
        """Stream LLM and tool outputs live."""
        # Inform the caller that the request is being sent to the LLM
        yield "[status] Sending prompt to LLM...\n"

        # Ensure output directory exists for debug logs
        try:
            os.makedirs("output", exist_ok=True)
            log_path = os.path.join("output", "stream_debug.log")
            log_file = open(log_path, "a", encoding="utf-8")
        except Exception:
            log_file = None

        for message_chunk, metadata in self.runnable.stream(
            input={"messages": [HumanMessage(content=message)]},
            stream_mode="messages",
            **kwargs,
        ):
            # Log the raw chunk and metadata for debugging (if possible)
            try:
                if log_file:
                    log_file.write("--- CHUNK ---\n")
                    log_file.write(repr(message_chunk) + "\n")
                    log_file.write("META: " + repr(metadata) + "\n")
                    log_file.flush()
            except Exception:
                pass
            if isinstance(message_chunk, AIMessageChunk):
                # Handle tool calls
                if message_chunk.response_metadata:
                    finish_reason = message_chunk.response_metadata.get(
                        "finish_reason", ""
                    )
                    if finish_reason == "tool_calls":
                        yield "\n\n"

                if message_chunk.tool_call_chunks:
                    tool_chunk = message_chunk.tool_call_chunks[0]
                    tool_name = tool_chunk.get("name", "")
                    args = tool_chunk.get("args", "")
                    if tool_name:
                        yield f"\n\n< TOOL CALL: {tool_name} >\n\n"
                        # Inform the user that the tool is being executed
                        yield f"[status] Executing tool: {tool_name} (this may take a few seconds)...\n"
                        # Attempt to execute the tool immediately and stream results
                        exec_result = self._execute_tool_call(tool_name, args)

                        # If exec_result is a Command-like object with update, try to return helpful info
                        if hasattr(exec_result, "update") and isinstance(
                            exec_result.update, dict
                        ):
                            update = exec_result.update
                            # If the agent saved a PNG, report the filename or indicate base64
                            png_path = update.get("chart_png_path")
                            png_b64 = update.get("chart_png_base64")
                            if png_path:
                                yield f"[status] Visualization generated -> {png_path}\n"
                            elif png_b64:
                                yield f"[status] Visualization generated (base64 PNG, {len(png_b64)} chars)\n"
                            else:
                                # fallback: try to find the JSON filename from args
                                filename = None
                                try:
                                    import json

                                    parsed = (
                                        json.loads(args)
                                        if isinstance(args, str)
                                        else args
                                    )
                                    if isinstance(parsed, dict) and parsed.get("name"):
                                        filename = f"output/{parsed.get('name')}.json"
                                except Exception:
                                    # ignore parse errors
                                    pass
                                if filename:
                                    yield f"[status] Visualization generated -> {filename}\n"
                                else:
                                    yield "[status] Visualization generated.\n"
                            continue

                        # If the tool stored a DataFrame in session.df, stream it in chunks
                        try:
                            df = getattr(session, "df", None)
                        except Exception:
                            df = None

                        if df is not None:
                            try:
                                total_rows = len(df)
                            except Exception:
                                total_rows = None

                            header = "[status] Tool returned a tabular result."
                            # CSV backups to disk have been disabled; report row count only
                            if total_rows is not None:
                                header = f"[status] Tool returned {total_rows} rows."

                            yield header + "\n\n"

                            # stream in chunks to avoid overwhelming the client
                            try:
                                chunk_size = 1000
                                for start in range(0, total_rows or 0, chunk_size):
                                    end = min(start + chunk_size, total_rows)
                                    chunk = df.iloc[start:end]
                                    yield f"-- rows {start+1} to {end} --\n"
                                    try:
                                        yield chunk.to_markdown(index=False) + "\n\n"
                                    except Exception:
                                        yield chunk.to_csv(index=False) + "\n\n"
                            except Exception:
                                # fallback: yield the exec_result fallback
                                yield str(exec_result) + "\n"

                            # clear session.df to avoid reusing the same data
                            try:
                                session.df = None
                            except Exception:
                                pass
                        else:
                            # Otherwise, stream the raw result or error
                            yield str(exec_result) + "\n"
                    if args:
                        yield args + "\n"
                else:
                    # ensure we yield a string
                    yield str(message_chunk.content)
            continue
        # Close log file if opened
        try:
            if log_file:
                log_file.close()
        except Exception:
            pass

    def _find_tool_callable(self, name: str):
        """Find a callable tool by name from self.tools.

        Tries common attributes: __name__, name, or repr matching.
        """
        for t in self.tools:
            # unwrap if it's a wrapper with .func
            candidate = getattr(t, "func", t)
            # check __name__
            if hasattr(candidate, "__name__") and candidate.__name__ == name:
                return candidate
            # check 'name' attribute
            if hasattr(candidate, "name") and getattr(candidate, "name") == name:
                return candidate
            # fallback: string match
            if name in repr(candidate):
                return candidate
        return None

    def _execute_tool_call(self, tool_name: str, args_text: str | None):
        """Execute a tool by name with args parsed from a string.

        Attempts JSON parse, then literal_eval. Returns the tool's result or an error string.
        """
        tool_callable = self._find_tool_callable(tool_name)
        if not tool_callable:
            return f"Error: tool '{tool_name}' not found."

        # Parse args_text into kwargs if possible
        parsed_args = None
        try:
            import json

            parsed_args = json.loads(args_text) if args_text is not None else None
        except Exception:
            try:
                from ast import literal_eval

                if args_text is not None:
                    parsed_args = literal_eval(args_text)
                else:
                    parsed_args = None
            except Exception:
                # If not dict-like, keep as raw string
                parsed_args = args_text

        try:
            # If parsed_args is a dict-like, call with kwargs. For tuples/lists, use positional.
            if isinstance(parsed_args, dict):
                # Ensure tool_call_id param exists for tools that expect it
                if "tool_call_id" not in parsed_args:
                    parsed_args["tool_call_id"] = "graph_direct"
                return tool_callable(**parsed_args)
            elif isinstance(parsed_args, (list, tuple)):
                # append a default tool_call_id
                return tool_callable(*parsed_args, tool_call_id="graph_direct")
            else:
                # single string arg (common case)
                return tool_callable(parsed_args)
        except Exception as e:
            return f"Error executing tool '{tool_name}': {e}"

    # -----------------------
    # Conversation persistence
    # -----------------------
    def _ensure_conversation_table(self):
        """Create the conversation_messages table if it doesn't exist."""
        create_sql = """
        CREATE TABLE IF NOT EXISTS conversation_messages (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT,
            created_at TIMESTAMPTZ DEFAULT now()
        )
        """
        with session.engine.connect() as conn:
            conn.execute(sa_text(create_sql))
            conn.commit()

    def _load_conversation_messages(self, session_id: str) -> list[BaseMessage]:
        """Load conversation messages for a session from the DB and map them to message objects.

        Returns a list of BaseMessage instances ordered by created_at ascending.
        """
        msgs: list[BaseMessage] = []
        try:
            # Try multiple possible column names (content / text / context)
            select_sql = "SELECT role, content FROM conversation_messages WHERE session_id = :session_id ORDER BY created_at"
            with session.engine.connect() as conn:
                result = conn.execute(sa_text(select_sql), {"session_id": session_id})
                rows = result.fetchall()

            import re

            def normalize_stored_text(raw: str) -> str:
                """Attempt to extract the most relevant user/assistant text from stored content.

                Handles cases where the assistant saved a formatted block like:
                "**Your question**\n> *“original question”*\n..."
                or other markdown wrappers. Falls back to the raw string.
                """
                if not raw:
                    return ""
                s = raw.strip()

                # If the content looks like JSON, try to parse and extract 'content'
                try:
                    import json

                    parsed = json.loads(s)
                    if isinstance(parsed, dict) and parsed.get("content"):
                        s = str(parsed.get("content"))
                except Exception:
                    pass

                # Remove fenced code blocks
                s = re.sub(r"```[\s\S]*?```", "", s)

                # Remove markdown headings and horizontal rules
                s = re.sub(r"^\s*#.*$", "", s, flags=re.MULTILINE)
                s = re.sub(r"^-{3,}$", "", s, flags=re.MULTILINE)

                # If the assistant stored a 'Your question' section, try to extract the quoted text
                m = re.search(
                    r"Your question[:\s\-]*\n[\s\>\*\"]*([^\n\r]+)",
                    s,
                    flags=re.IGNORECASE,
                )
                if m:
                    candidate = m.group(1).strip()
                    # remove leading blockquote marker '>' and surrounding quotes/stars
                    candidate = re.sub(r"^[>\s\*\"]+", "", candidate)
                    candidate = re.sub(r"[\"\'\*]+$", "", candidate)
                    return candidate

                # Look for blockquote style lines starting with '>' and take first non-empty
                for line in s.splitlines():
                    line = line.strip()
                    if line.startswith(">"):
                        text = line.lstrip("> ").strip()
                        if text:
                            return text

                # Try to find smart-quoted or straight-quoted substring
                q = re.search(r"[“\"]([^”\"]{5,2000})[”\"]", s)
                if q:
                    return q.group(1).strip()

                # Fallback: return the first non-empty line
                for line in s.splitlines():
                    if line.strip():
                        return line.strip()

                return s

            for role, content in rows:
                if content is None:
                    content = ""
                try:
                    norm = normalize_stored_text(str(content))
                    if role == "user":
                        msgs.append(HumanMessage(content=norm))
                    elif role == "assistant":
                        msgs.append(AIMessage(content=norm))
                    elif role == "system":
                        msgs.append(SystemMessage(content=norm))
                    else:
                        msgs.append(AIMessage(content=norm))
                except Exception:
                    msgs.append(HumanMessage(content=str(content)))
        except Exception:
            # On any error, return empty history
            return []
        return msgs

    def _save_message(self, session_id: str, role: str, content: str):
        """Insert a message row into conversation_messages."""
        if content is None:
            content = ""
        insert_sql = "INSERT INTO conversation_messages (id, session_id, role, content, created_at) VALUES (:id, :session_id, :role, :content, now())"
        params = {
            "id": str(uuid.uuid4()),
            "session_id": session_id,
            "role": role,
            "content": content,
        }
        with session.engine.connect() as conn:
            conn.execute(sa_text(insert_sql), params)
            conn.commit()


# -----------------------
# Instantiate the agent
# -----------------------
agent = Agent(name="Scout", system_prompt=prompts.scout_system_prompt)
graph = agent.build_graph()


if __name__ == "__main__":
    # Safe smoke-test when running the module directly. Avoid calling the LLM or tools.
    print(f"Agent name: {agent.name}")
    try:
        tool_names = [t.__name__ for t in agent.tools]
    except Exception:
        # tools might be wrapped; fall back to repr
        tool_names = [repr(t) for t in agent.tools]
    print(f"Tools available: {tool_names}")
    print(f"Runnable type: {type(graph)}")
    print(
        "Smoke-test ready. To run interactions, call agent.invoke(message) or use the LangGraph runtime."
    )

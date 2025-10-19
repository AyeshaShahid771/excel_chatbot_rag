"""
FastAPI with per-user sessions using cookies.
Each browser/user gets a unique session ID that persists across page reloads.
New session created only when cookie is deleted or browser cleared.
Includes chat history management.
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from agent import graph
    agent = graph.agent
except Exception as e:
    print(f"Failed to import agent.graph: {e}")
    agent = None


app = FastAPI(
    title="Data Analyst Agent API",
    description="Expert data analyst AI agent with per-user sessions"
)

# Enable CORS for browser requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSION_COOKIE_NAME = "agent_session_id"
SESSION_COOKIE_MAX_AGE = 7 * 24 * 60 * 60  # 7 days

# In-memory storage for chat history: {session_id: [messages]}
chat_history: Dict[str, List[dict]] = {}


class ChatRequest(BaseModel):
    message: str
    stream: bool = False
    session_id: str | None = None


class ChatMessage(BaseModel):
    id: str
    user_message: str
    agent_reply: str
    timestamp: str


def get_or_create_session(request: Request, response: Response, explicit_session_id: str | None = None) -> str:
    """
    Get session ID from cookie, or create a new one if it doesn't exist.
    Sets/updates the cookie in the response.
    """
    # Prefer an explicit session_id supplied by the client (body/query) to support non-browser clients
    session_id = explicit_session_id or request.cookies.get(SESSION_COOKIE_NAME)

    if not session_id:
        session_id = str(uuid.uuid4())
        print(f"🆕 New user session created: {session_id}")
        chat_history[session_id] = []
    else:
        print(f"♻️  Existing user session found: {session_id}")
    
    # Set cookie (will persist across page reloads)
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_id,
        max_age=SESSION_COOKIE_MAX_AGE,
        httponly=True,
        secure=False,  # Set to True if using HTTPS
        samesite="lax"
    )
    
    return session_id


@app.get("/")
async def root(request: Request, response: Response):
    """Root endpoint with API information"""
    # Create or retrieve session from cookie (no request body available on GET)
    session_id = get_or_create_session(request, response)
    
    return {
        "message": "Hi, I am an expert data analyst AI agent",
        "description": "I'm here to assist you in analysis of your three tables: brand_proposals, brand_payments, brand_contacts",
        "session_id": session_id,
        "endpoints": {
            "chat": "POST /chat - Non-streaming response",
            "health": "GET /health - Health check",
            "create_new_chat": "POST /create_new_chat - Create a new session",
            "chat_history": "GET /chat-history - Get all messages for current session",
            "delete_chat_history": "DELETE /chat-history - Clear all messages for current session"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if agent is None:
        return {"status": "error", "agent": "not initialized"}
    return {"status": "ok", "agent": "ready"}


@app.post("/chat")
async def chat(req: ChatRequest, request: Request, response: Response):
    """
    Chat endpoint with session persistence.
    Same user gets same session across page reloads.
    """
    if agent is None:
        response.status_code = 503
        return {"error": "Agent not initialized"}
    
    if not req.message.strip():
        response.status_code = 400
        return {"error": "Message cannot be empty"}
    
    session_id = get_or_create_session(request, response)
    
    try:
        if req.stream:
            output = ""
            for chunk in agent.stream(req.message, session_id=session_id):
                output += str(chunk)
            
            agent_reply = output.strip()
        else:
            resp = agent.invoke(req.message, session_id=session_id)
            agent_reply = str(resp)
        
    # Store message in chat history
        message_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        chat_entry = {
            "id": message_id,
            "user_message": req.message,
            "agent_reply": agent_reply,
            "timestamp": timestamp
        }
        
        if session_id not in chat_history:
            chat_history[session_id] = []
        
        chat_history[session_id].append(chat_entry)
        
        return {
            "session_id": session_id,
            "user_message": req.message,
            "agent_reply": agent_reply,
            "stream": req.stream,
            "id": message_id,
            "timestamp": timestamp
        }
    
    except Exception as e:
        response.status_code = 500
        return {
            "error": f"Error invoking agent: {str(e)}",
            "session_id": session_id
        }


@app.get("/chat-history")
async def get_chat_history(request: Request, response: Response, session_id: str | None = None):
    """
    Retrieve all chat messages for the current session.
    """
    # Prefer explicit session_id query param, then cookie
    session_id = session_id or request.cookies.get(SESSION_COOKIE_NAME)
    if not session_id:
        # create a new session if none supplied
        session_id = get_or_create_session(request, response)
    
    if session_id not in chat_history:
        chat_history[session_id] = []
    
    return {
        "session_id": session_id,
        "messages": chat_history[session_id]
    }


@app.delete("/chat-history")
async def delete_chat_history(request: Request, response: Response, session_id: str | None = None):
    """
    Delete all chat messages for the current session.
    """
    session_id = session_id or request.cookies.get(SESSION_COOKIE_NAME)
    if not session_id:
        session_id = get_or_create_session(request, response)
    
    if session_id in chat_history:
        chat_history[session_id] = []
        print(f"🗑️  Chat history cleared for session: {session_id}")
    
    return {
        "message": "Chat history cleared",
        "session_id": session_id
    }


@app.post("/create_new_chat")
async def create_new_chat(response: Response):
    """
    Create a new chat session for the user (clears the previous session cookie).
    """
    response.delete_cookie(SESSION_COOKIE_NAME)
    new_session_id = str(uuid.uuid4())
    
    # Initialize empty chat history for new session
    chat_history[new_session_id] = []
    
    # Set new cookie
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=new_session_id,
        max_age=SESSION_COOKIE_MAX_AGE,
        httponly=True,
        secure=False,
        samesite="lax"
    )
    
    print(f"🔄 New chat session created: {new_session_id}")
    
    return {
        "message": "New chat session created.",
        "new_session_id": new_session_id
    }


@app.on_event("startup")
def startup_event():
    """Log startup info"""
    if agent:
        print("✓ Agent initialized successfully")
    else:
        print("✗ Agent failed to initialize")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=3000)   
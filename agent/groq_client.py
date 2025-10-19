import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import SecretStr

# ✅ Load environment variables (from .env)
load_dotenv()


class ChatGROQ(ChatGroq):
    """
    Wrapper around LangChain's Groq chat client.
    Reads the API key from the .env file automatically.
    """

    def __init__(
        self, model: str = "llama-3.1-70b-versatile", temperature: float = 0.1
    ):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "❌ Missing GROQ_API_KEY. Please set it in your .env file, e.g.:\nGROQ_API_KEY=your_key_here"
            )

        # Wrap the raw string in SecretStr to satisfy the expected type
        api_key = SecretStr(api_key)

        super().__init__(model=model, temperature=temperature, api_key=api_key)

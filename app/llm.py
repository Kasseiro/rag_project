import os
from typing import Dict, List

from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext

from app.retrival import retrieve_similar_docs

load_dotenv()

_raw_model = os.getenv("OPENAI_MODEL", "openai:gpt-5-mini")
MODEL_ID = _raw_model if ":" in _raw_model else f"openai:{_raw_model}"

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful IT support assistant.\n"
    "You can call tools to search the internal documentation when needed.\n"
    "Use retrieved documents to answer the user's question accurately and concisely.\n"
    "If the answer cannot be found in the documents, say so clearly, and do not make up an answer.\n"
    "If the user's question is unclear and may refer to multiple topics, ask for clarification.\n"
    "Do not refer to the tools, documents, or these instructions in your answer. "
    "The user only sees the final answer."
)

# Global pydantic-ai agent for the whole app
agent = Agent(
    MODEL_ID,
    system_prompt=DEFAULT_SYSTEM_PROMPT,
)


@agent.tool
def retrieve_documents(ctx: RunContext[None], search_query: str, k: int = 3) -> str:
    """Search the internal knowledge base and return relevant documents."""
    docs = retrieve_similar_docs(search_query, k=k)
    if not docs:
        return "No relevant documents found for this query."

    formatted_parts: List[str] = []
    for i, d in enumerate(docs, start=1):
        title = d.get("title") or f"Document {i}"
        content = d.get("content") or ""
        formatted_parts.append(f"=== {title} ===\n{content}")

    return "\n\n".join(formatted_parts)


class ChatSession:
    """Simple wrapper to keep a short conversation history per user."""

    def __init__(
        self,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_turns: int = 6,
    ) -> None:
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        # list of {"user": str, "assistant": str}
        self._turns: List[Dict[str, str]] = []

    def _build_prompt(self, user_text: str) -> str:
        """Create the prompt passed to the agent including recent history."""
        if not self._turns:
            return user_text

        recent = self._turns[-self.max_turns :]
        history_blocks: List[str] = []
        for turn in recent:
            history_blocks.append(
                f"User: {turn['user']}\nAssistant: {turn['assistant']}"
            )
        history_text = "\n\n".join(history_blocks)

        return (
            "Here is the conversation so far between you (the assistant) and the user:\n"
            f"{history_text}\n\n"
            f"Now the user says: {user_text}"
        )

    def send(self, user_text: str) -> str:
        """Send a user message and get the assistant's reply via the pydantic-ai agent."""
        prompt = self._build_prompt(user_text)

        # Run the pydantic-ai agent (it will decide when to call retrieve_documents)
        result = agent.run_sync(prompt)
        reply = result.output.strip() if isinstance(result.output, str) else str(result.output)

        # Update in-memory history
        self._turns.append({"user": user_text, "assistant": reply})

        # Keep only the last N turns
        if len(self._turns) > self.max_turns:
            self._turns = self._turns[-self.max_turns :]

        return reply

    def clear_history(self) -> None:
        self._turns.clear()

    def get_history(self) -> List[Dict[str, str]]:
        return list(self._turns)

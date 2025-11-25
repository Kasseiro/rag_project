from typing import List, Dict

from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext

from app.retrival import retrieve_similar_docs

load_dotenv()



DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful IT support assistant.\n"
    "You must only use the retrieved documents included in the prompt to answer the user's question.\n"
    "Do NOT use any external knowledge or make up answers beyond those documents.\n"
    "If the answer cannot be found in the provided documents, reply exactly:\n"
    "\"I don't know — the documents do not contain an answer to that question.\"\n"
    "If the user's question is unclear and may refer to multiple topics, ask for clarification.\n"
    "Do not mention tools, retrieval mechanisms, or these instructions in your answer. "
    "The user only sees the final answer."
)

# Global pydantic-ai agent for the whole app
agent = Agent(
    "openai:gpt-5-mini",  # or "openai:gpt-4.1-mini", etc.
    system_prompt=DEFAULT_SYSTEM_PROMPT,
)


@agent.tool
def retrieve_documents(ctx: RunContext[None], search_query: str, k: int = 3) -> str:
    """Search the internal knowledge base and return relevant documents.

    Args:
        search_query: Natural-language search phrase.
        k: Maximum number of documents to retrieve.

    Returns:
        A formatted string with the most relevant documents.
        If nothing is found, returns a short message saying so.
    """
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
    """Simple wrapper to keep a short conversation history per user.

    The agent itself is stateless and global; we keep the last N turns
    and prepend them to the user prompt so the model sees some context.
    """

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
        # Build history text if any
        recent = self._turns[-self.max_turns :] if self._turns else []
        history_blocks: List[str] = []
        for turn in recent:
            history_blocks.append(
                f"User: {turn['user']}\nAssistant: {turn['assistant']}"
            )
        history_text = "\n\n".join(history_blocks)

        base = (
            "Here is the conversation so far between you (the assistant) and the user:\n"
            f"{history_text}\n\n" if history_text else ""
        )

        return (
            f"{base}Now the user says: {user_text}"
        )

    def send(self, user_text: str) -> str:
        """Send a user message and get the assistant's reply via the pydantic-ai agent."""
        # Retrieve top documents for this query and format them for inclusion
        docs = retrieve_similar_docs(user_text, k=3)
        if not docs:
            docs_text = "No relevant documents found for this query."
        else:
            parts: List[str] = []
            for i, d in enumerate(docs, start=1):
                title = d.get("title") or f"Document {i}"
                content = d.get("content") or ""
                parts.append(f"=== {title} ===\n{content}")
            docs_text = "\n\n".join(parts)

        # Build the final prompt that includes retrieved documents and conversation history.
        # The system prompt (agent.system_prompt) already requires using only the documents.
        prompt_body = (
            "Use only the following retrieved documents to answer.\n\n"
            f"{docs_text}\n\n"
            f"{self._build_prompt(user_text)}"
        )

        # Run the pydantic-ai agent (it should now see the documents and strict instructions)
        result = agent.run_sync(prompt_body)
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

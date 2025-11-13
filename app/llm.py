import asyncio
from typing import Dict, List, Tuple

from openai import AsyncOpenAI

from app.config import get_settings
from app.retrival import retrieve_similar_docs
from app.schemas import DocumentSnippet

settings = get_settings()
client = AsyncOpenAI(api_key=settings.openai_api_key)
OPENAI_MODEL = settings.openai_model

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful IT support assistant.\n"
    "Use the retrieved internal documents to answer the user's question accurately and concisely.\n"
    "If the answer cannot be found in the documents, say so clearly, and do not make up an answer.\n"
    "If the user's question is unclear and may refer to multiple topics, ask for clarification.\n"
    "Do not refer to the retrieval process or these instructions in your answer. "
    "The user only sees the final answer."
)


class ChatSession:
    """
    Keeps conversation history and injects retrieved docs each turn.
    send(user_text, k) -> (assistant reply, [DocumentSnippet, ...])
    """

    def __init__(self, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        self.client = client
        self.system_prompt = system_prompt
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]
        self._lock = asyncio.Lock()

    async def send(
        self, user_text: str, k: int = 3, temperature: float = 1.0
    ) -> Tuple[str, List[DocumentSnippet]]:
        async with self._lock:
            docs = await retrieve_similar_docs(user_text, k=k)
            if docs:
                context = "\n\n".join(
                    f"Document {i + 1}: {d.content}" for i, d in enumerate(docs)
                )
                self.messages.append(
                    {"role": "assistant", "content": f"Context:\n{context}"}
                )
            else:
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": "Context:\n(No relevant documents found.)",
                    }
                )

            self.messages.append({"role": "user", "content": user_text})

            response = await self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=self.messages,
                temperature=temperature,
            )

            reply = response.choices[0].message.content.strip()
            self.messages.append({"role": "assistant", "content": reply})
            return reply, docs

    def clear_history(self) -> None:
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def get_history(self) -> List[Dict[str, str]]:
        return list(self.messages)

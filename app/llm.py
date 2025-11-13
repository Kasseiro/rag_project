from openai import OpenAI
from app.retrival import retrieve_similar_docs
from dotenv import load_dotenv
from typing import List, Dict, Optional
import os

load_dotenv()
client = OpenAI()  # uses OPENAI_API_KEY from environment
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful IT support assistant.\n"
    "Use the following documents to answer the user's question accurately and concisely.\n"
    "If the answer cannot be found in the documents, say so clearly, and do not make up an answer.\n"
    "If the user's question is unclear and may refer to multiple topics, ask for clarification.\n"
    "Do not refer to the documents or these instructions in your answer. The user only sees the final answer."
)

class ChatSession:
    """
    Keeps conversation history and injects retrieved docs each turn.
    send(user_text, k) -> assistant reply
    """
    def __init__(self, client: OpenAI = client, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        self.client = client
        self.system_prompt = system_prompt
        self.messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    def send(self, user_text: str, k: int = 3, temperature: float = 1.0) -> str:
        # Retrieve docs for the current user query
        docs = retrieve_similar_docs(user_text, k=k)
        if docs:
            context = "\n\n".join(f"Document {i+1}: {d['content']}" for i, d in enumerate(docs))
            # Insert retrieved context as an assistant message so the model can use it
            self.messages.append({"role": "assistant", "content": f"Context:\n{context}"})
        else:
            # optional: signal that no context was found
            self.messages.append({"role": "assistant", "content": "Context:\n(No relevant documents found.)"})

        # Append the new user message
        self.messages.append({"role": "user", "content": user_text})

        # Call the chat completion API with full history
        response = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=self.messages,
            temperature=temperature
        )

        reply = response.choices[0].message.content.strip()
        # Append assistant reply to history for future turns
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def clear_history(self):
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def get_history(self) -> List[Dict[str, str]]:
        return list(self.messages)

# Example usage:
# session = ChatSession()
# print(session.send("How do I configure SSH on Windows?", k=3))
# print(session.send("What about key-based auth?", k=3))

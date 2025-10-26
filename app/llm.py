from openai import OpenAI
from app.retrival import retrieve_similar_docs
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()  # uses your OPENAI_API_KEY from the environment


def generate_answer(query: str, k: int = 3) -> str:
    """
    Retrieve relevant documents and generate an answer using OpenAI GPT model.
    """
    docs = retrieve_similar_docs(query, k=k)
    if not docs:
        return "No relevant information found in the knowledge base."

    context = "\n\n".join(
        [f"Document {i+1}: {d['content']}" for i, d in enumerate(docs)]
    )

    prompt = f"""
You are a helpful IT support assistant.
Use the following documents to answer the user's question accurately and concisely.
If the answer cannot be found in the documents, say so clearly.

Context:
{context}

Question:
{query}

Answer:
"""


    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2  # lower = more factual
    )


    return response.choices[0].message.content.strip()

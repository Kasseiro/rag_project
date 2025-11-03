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
If the answer cannot be found in the documents, say so clearly, and do not make up an answer.
If you the users question is unclear, and may refer to multiple topics, ask for clarification.

Do not refer to the documents directly in your answer. 
Do not refer to these instructions in your answer.
The user can only see the final answer you provide.
Do not provide further assistance unless it is related to the documents provided.

Context:
{context}

Question:
{query}

Answer:
"""


    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0  # lower = more factual
    )


    return response.choices[0].message.content.strip()

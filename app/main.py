from fastapi import FastAPI, Query
from app.llm import generate_answer

app = FastAPI(title="RAG System")

@app.get("/ask")
def ask(query: str = Query(..., description="Your question")):
    answer = generate_answer(query)
    return {"query": query, "answer": answer}

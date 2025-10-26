from app.llm import generate_answer

query = "How do I reset my password?"
answer = generate_answer(query)

print("\n Final RAG Answer:\n")
print(answer)

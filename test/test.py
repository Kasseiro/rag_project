from app.retrival import retrieve_similar_docs

query = "How do I reset my password?"
docs = retrieve_similar_docs(query, k=3)

print(f"\n Query: {query}\n")
if not docs:
    print("❌ No results found.")
else:
    for d in docs:
        print(f"Title: {d['title']}")
        print(f"Score: {d['distance']:.4f}")
        print(f"Text: {d['content'][:300]}...\n")

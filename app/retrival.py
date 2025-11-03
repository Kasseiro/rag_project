from app.db import SessionLocal
from openai import OpenAI
from sqlalchemy import text
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()


def retrieve_similar_docs(query: str, k: int = 3):

    session = SessionLocal()
    try:
        query_embedding = client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        ).data[0].embedding

        sql = text("""
            SELECT id, title, content, embedding <-> (:query_embedding)::vector AS distance
            FROM documents
            ORDER BY distance ASC
            LIMIT :k;
        """)

        results = session.execute(sql, {"query_embedding": query_embedding, "k": k}).fetchall()

        docs = []
        for row in results:
            docs.append({
                "id": row.id,
                "title": row.title,
                "content": row.content,
                "distance": row.distance
            })

        return docs

    except Exception as e:
        print(f"❌ Retrieval error: {e}")
        return []

    finally:
        session.close()











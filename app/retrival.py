from typing import List
from app.db import SessionLocal
from openai import OpenAI
from sqlalchemy import text
from app.config import get_settings
from app.schemas import DocumentSnippet

settings = get_settings()
client = OpenAI(api_key=settings.openai_api_key)
EMBEDDING_MODEL = settings.openai_embedding_model


def retrieve_similar_docs(query: str, k: int = 3) -> List[DocumentSnippet]:
    session = SessionLocal()
    try:
        query_embedding = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=query
        ).data[0].embedding

        sql = text("""
            SELECT id, title, content, embedding <-> (:query_embedding)::vector AS distance
            FROM documents
            ORDER BY distance ASC
            LIMIT :k;
        """)

        results = session.execute(sql, {"query_embedding": query_embedding, "k": k}).fetchall()

        docs = [
            DocumentSnippet(
                id=row.id,
                title=row.title,
                content=row.content,
                distance=float(row.distance),
            )
            for row in results
        ]

        return docs

    except Exception as e:
        print(f"❌ Retrieval error: {e}")
        return []

    finally:
        session.close()










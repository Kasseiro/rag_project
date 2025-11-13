from typing import List
from sqlalchemy import text
from openai import AsyncOpenAI
from app.db import AsyncSessionLocal
from app.config import get_settings
from app.schemas import DocumentSnippet

settings = get_settings()
client = AsyncOpenAI(api_key=settings.openai_api_key)
EMBEDDING_MODEL = settings.openai_embedding_model


async def retrieve_similar_docs(query: str, k: int = 3) -> List[DocumentSnippet]:
    async with AsyncSessionLocal() as session:
        try:
            embedding_response = await client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=query
            )
            query_embedding = embedding_response.data[0].embedding

            sql = text("""
                SELECT id, title, content, embedding <-> (:query_embedding)::vector AS distance
                FROM documents
                ORDER BY distance ASC
                LIMIT :k;
            """)

            result = await session.execute(sql, {"query_embedding": query_embedding, "k": k})
            rows = result.fetchall()

            docs = [
                DocumentSnippet(
                    id=row.id,
                    title=row.title,
                    content=row.content,
                    distance=float(row.distance or 0.0),
                )
                for row in rows
            ]

            return docs

        except Exception as e:
            print(f"❌ Retrieval error: {e}")
            return []










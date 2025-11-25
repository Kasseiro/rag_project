from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy import text

from app.db import SessionLocal

load_dotenv()

# Single global client for embeddings
_client = OpenAI()


def retrieve_similar_docs(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """Return top-k similar documents from Postgres (pgvector).

    Args:
        query: user query text
        k: number of docs to return

    Returns:
        list of dicts with keys: {"id", "title", "content", "distance"}.
        Returns [] on error.
    """
    session = SessionLocal()
    try:
        # create embedding for the query
        embedding_resp = _client.embeddings.create(
            model="text-embedding-3-small",
            input=query,
        )
        query_embedding = embedding_resp.data[0].embedding
        # serialize to Postgres vector literal "[0.1,0.2,...]"
        query_embedding_param = f"[{','.join(map(str, query_embedding))}]"

        # Vector similarity search
        sql = text(
            """
            SELECT
                id,
                title,
                content,
                embedding <-> (:query_embedding)::vector AS distance
            FROM documents
            ORDER BY distance ASC
            LIMIT :k;
            """
        )

        result = session.execute(
            sql,
            {
                "query_embedding": query_embedding_param,
                "k": k,
            },
        )

        rows = result.fetchall()
        docs: List[Dict[str, Any]] = []
        for row in rows:
            docs.append(
                {
                    "id": row.id,
                    "title": getattr(row, "title", None),
                    "content": getattr(row, "content", None),
                    "distance": getattr(row, "distance", None),
                }
            )

        return docs

    except Exception as exc:
        # In a real app, replace this with proper logging
        print(f"[retrieve_similar_docs] Error while searching docs: {exc}")
        return []

    finally:
        session.close()

from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy import text

from app.db import SessionLocal

load_dotenv()

# Single global client for embeddings
_client = OpenAI()


def retrieve_similar_docs(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """Retrieve the top-k most similar documents from Postgres using pgvector.

    Args:
        query: Natural-language search query.
        k: Number of documents to return.

    Returns:
        A list of dicts: {"id", "title", "content", "distance"}.
        Returns an empty list on error.
    """
    session = SessionLocal()
    try:
        # Create embedding for the query
        embedding_resp = _client.embeddings.create(
            model="text-embedding-3-small",
            input=query,
        )
        query_embedding = embedding_resp.data[0].embedding
        # Serialize the embedding as a Postgres vector literal like
        # `[0.1,0.2,...]` because some DB drivers serialize Python lists
        # as Postgres arrays (`{...}`) which do not cast to `vector`.
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

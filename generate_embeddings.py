from openai import OpenAI
from app.db import SessionLocal
from app.models import Document
from app.config import get_settings

settings = get_settings()
client = OpenAI(api_key=settings.openai_api_key)

session = SessionLocal()

try:
    docs = session.query(Document).filter(Document.embedding == None).all()
    print(f"Found {len(docs)} documents to embed.")

    # 2️⃣ Generate embeddings
    for i, doc in enumerate(docs, start=1):
        response = client.embeddings.create(
            model=settings.openai_embedding_model,
            input=doc.content
        )
        doc.embedding = response.data[0].embedding  # list of floats

        if i % 100 == 0:
            session.commit()
            print(f"✅ Committed {i} embeddings...")

    session.commit()
    print("✅ All embeddings generated and saved successfully.")

except Exception as e:
    session.rollback()
    print(f"❌ Error: {e}")

finally:
    session.close()

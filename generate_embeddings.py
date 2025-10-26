from sentence_transformers import SentenceTransformer
from app.db import SessionLocal
from app.models import Document

model = SentenceTransformer('all-MiniLM-L6-v2')

session = SessionLocal()

try:
    docs = session.query(Document).filter(Document.embedding == None).all()
    for doc in docs:
        embedding = model.encode(doc.content).tolist()
        doc.embedding = embedding

    session.commit()
    print("✅ All embeddings generated and saved successfully.")

except Exception as e:
    session.rollback()
    print(f"An error occurred: {e}")

finally:
    session.close()
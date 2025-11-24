from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from app.models import Base
from dotenv import load_dotenv
import os

# Load environment variables from the .env file (if present)
load_dotenv()

# Access environment variables as if they came from the actual environment
DATABASE_URL = os.getenv('DATABASE_URL')

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Ensure pgvector extension exists, then create tables.

    Note: creating extensions requires appropriate DB privileges (usually
    a superuser). If the current DB user lacks permission the extension
    creation will fail but table creation will still be attempted.
    """
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
    except Exception as exc:
        # Avoid hard failure here; surface a warning so developers know
        # to run the command manually or use a DB superuser.
        print(f"[init_db] Warning: could not create pgvector extension: {exc}")

    Base.metadata.create_all(bind=engine)
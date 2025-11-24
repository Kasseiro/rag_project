# rag_project

Basic rag systmes that uses an modiefied it helpdesk dataset from: https://www.kaggle.com/datasets/dkhundley/synthetic-it-related-knowledge-items 


Commands for setup:

docker-compose up --build

docker-compose exec backend python -c "from app.db import init_db; init_db()"

docker-compose exec backend python insert_data.py

docker-compose exec backend python generate_embeddings.py



Exampel of .env file: 

OPENAI_API_KEY= ..... 

DATABASE_URL="postgresql+psycopg2://user:password@db:5432/ragdb"


Runs on : http://localhost:8000/ 

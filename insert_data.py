import pandas as pd
from app.db import SessionLocal
from app.models import Document


df = pd.read_csv('data/prepared_dataset.csv')

session = SessionLocal()
for _, row in df.iterrows():
    doc = Document(title=row['title'], content=row['content'])
    session.add(doc)


session.commit()
session.close()
print("Data inserted successfully.")
# rag_project

Basic rag systmes that uses an modiefied it helpdesk dataset from: https://www.kaggle.com/datasets/dkhundley/synthetic-it-related-knowledge-items 

## Environment configuration

1. Start by copying the provided `.env.example` file to `.env`:

   ```bash
   cp .env.example .env
   ```

2. Update the variables in `.env` to match your local setup. The same file is read by Docker Compose for database container settings and by the backend for its application configuration.

3. Set `OPENAI_API_KEY` and `DATABASE_URL` in `.env`; without them the backend cannot reach OpenAI or the database. Optionally override `OPENAI_MODEL` if you need a different model than the default `gpt-5-mini`.

4. Before deploying to production, review the copied `.env` and replace the sample values (database credentials, connection strings, container names, exposed ports, etc.) with strong secrets and the correct production-specific values. Keep the `.env` file out of source control.


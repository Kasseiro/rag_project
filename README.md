# rag_project

Basic rag systmes that uses an modiefied it helpdesk dataset from: https://www.kaggle.com/datasets/dkhundley/synthetic-it-related-knowledge-items 

## Environment configuration

1. Start by copying the provided `.env.example` file to `.env`:

   ```bash
   cp .env.example .env
   ```

2. Update the variables in `.env` to match your local setup. The same file is read by Docker Compose for container configuration and by the backend for its application settings.

3. Set `OPENAI_API_KEY` and `DATABASE_URL` in `.env`; without them the backend cannot reach OpenAI or the database. Optionally override `OPENAI_MODEL` or `OPENAI_EMBEDDING_MODEL` if you need non-default model names.

4. Before deploying to production, review the copied `.env` and replace the sample values (database credentials, connection strings, container names, exposed ports, etc.) with strong secrets and the correct production-specific values. Keep the `.env` file out of source control.

> Note: The Postgres service is not exposed to the host; only the backend container can reach it on the internal Docker network. Expose the port manually only if you have a specific operational need.

## API usage

The `/ask` endpoint now accepts `POST` requests with a JSON body:

```json
{
  "query": "How do I reset my VPN token?",
  "top_k": 3,
  "temperature": 1.0
}
```

`top_k` and `temperature` are optional; defaults from the Pydantic model will be applied if you omit them. The response includes the answer plus the retrieved document snippets used for context.

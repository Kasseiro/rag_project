# rag_project

Basic rag system that uses a modified IT helpdesk dataset from: https://www.kaggle.com/datasets/dkhundley/synthetic-it-related-knowledge-items

## Environment configuration

1. Copy `.env.example` to `.env`:

   ```bash
   cp .env.example .env
   ```

2. Update the variables in `.env` for your environment. Docker Compose uses the same file for container configuration and the backend reads it for application settings.

3. Set `OPENAI_API_KEY` and `DATABASE_URL`; without them the backend cannot reach OpenAI or Postgres. Override `OPENAI_MODEL` or `OPENAI_EMBEDDING_MODEL` if you want non-default models.

4. Before deploying to production, replace the sample creds, connection strings, and container names with the correct production values. Keep `.env` out of source control.

> Note: The Postgres service is not exposed outside the Docker network; only the backend container can reach it. Publish the port manually only if required for operations.

## API usage

The `/ask` endpoint accepts a `POST` request with JSON:

```json
{
  "query": "How do I reset my VPN token?",
  "top_k": 3,
  "temperature": 1.0
}
```

`top_k` and `temperature` are optional — defaults from the Pydantic schema apply if omitted. The response contains the generated answer plus the retrieved document snippets that were injected into the LLM context.

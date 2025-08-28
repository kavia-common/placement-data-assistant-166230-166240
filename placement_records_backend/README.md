# Connecto (Beta) — Placement Records Backend

FastAPI backend implementing a strict RAG chatbot for historical placement records.

Features:
- REST API endpoints:
  - GET / — Health check
  - POST /ingest — Ingest placement records (embeds with llama-text-embed-v2 via OpenRouter and upserts into Pinecone)
  - POST /chat — Ask questions; retrieves top K=3 contexts from Pinecone and generates grounded answers via OpenRouter gpt-oss-20b (or configured model)
- Strict context policy: answers ONLY from provided placement records; otherwise returns fallback:
  "I don’t have information about that in the placement records."
- Open access (no authentication)
- Uses .env for all API keys and configuration

## Getting Started

1) Create .env
Copy `.env.example` to `.env` and provide real values.

Required keys:
- OPENROUTER_API_KEY
- PINECONE_API_KEY
- PINECONE_INDEX_NAME
- PINECONE_HOST (direct host for your Pinecone index, v2)

Optionally:
- SITE_URL (used for headers and docs)
- PINECONE_ENVIRONMENT / PINECONE_PROJECT_ID (legacy info, not required if PINECONE_HOST provided)

2) Install dependencies
The container already includes requirements.txt suitable for FastAPI uvicorn runtime.

3) Run
Use your preferred method to start the FastAPI app, e.g.:
uvicorn src.api.main:app --host 0.0.0.0 --port 3001 --reload

Docs: http://localhost:3001/docs

4) Generate OpenAPI spec file (optional)
python -m src.api.generate_openapi
This writes interfaces/openapi.json

## API

- GET /
Response: {"status": "ok"}

- POST /ingest
Body:
{
  "records": [
    {"id": "opt-1", "text": "Company ABC hired 15 students with avg CTC 8 LPA.", "metadata": {"company": "ABC", "year": 2023}}
  ]
}
Requires .env configured and Pinecone index host set.

- POST /chat
Body:
{
  "query": "How many students did ABC hire in 2023?",
  "top_k": 3
}
Response includes `answer`, `contexts`, and `used_fallback`.

## Notes

- Embeddings: llama-text-embed-v2 via OpenRouter embeddings endpoint.
- Generation: OpenRouter (model configurable). Default logic attempts a robust open model if the generic name is not found.
- Pinecone: This implementation uses direct REST calls and expects `PINECONE_HOST` for your index endpoint.

## RAG Policy (Strict)

- Use only retrieved context.
- If no/low-confidence context: return
  "I don’t have information about that in the placement records."
- 1–3 sentence answers, friendly and concise.
- No hallucinations.

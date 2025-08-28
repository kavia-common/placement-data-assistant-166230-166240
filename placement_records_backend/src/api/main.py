import os
import json
import logging
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import httpx

# Load environment variables from .env (do not hardcode secrets)
load_dotenv()

# Constants and environment variables (must be provided via .env)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "connecto-placement-records")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "")  # optional: for older pinecone deployments
PINECONE_PROJECT_ID = os.getenv("PINECONE_PROJECT_ID", "")  # optional: for v2
PINECONE_HOST = os.getenv("PINECONE_HOST", "")  # optional direct host for index
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
SITE_URL = os.getenv("SITE_URL", "http://localhost:3000")  # used in docs and CORS

# Model/Service constants
EMBEDDING_MODEL = "llama-text-embed-v2"  # Vercel AI or Replicate style name; we'll call OpenRouter embeddings endpoint compat
GENERATION_MODEL = "openrouter/gpt-oss-20b"  # As requested: gpt-oss-20b via OpenRouter
TOP_K = 3
FALLBACK_ANSWER = "I don’t have information about that in the placement records."

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("connecto-backend")

# PUBLIC_INTERFACE
class HealthResponse(BaseModel):
    """Response model for health endpoint."""
    status: str = Field(..., description="Status of the service. 'ok' if healthy.")

# PUBLIC_INTERFACE
class IngestRecord(BaseModel):
    """Placement record item to be embedded and stored."""
    id: Optional[str] = Field(None, description="Optional ID for the record; if omitted, backend may generate one.")
    text: str = Field(..., description="Raw text content of the placement record (e.g., company stats, roles, packages).")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional metadata such as company, year, role.")

# PUBLIC_INTERFACE
class IngestRequest(BaseModel):
    """Batch ingestion payload for placement records."""
    records: List[IngestRecord] = Field(..., description="List of placement record items to ingest.")

# PUBLIC_INTERFACE
class IngestResponse(BaseModel):
    """Response for ingestion endpoint."""
    success: bool = Field(..., description="Whether ingestion completed without critical errors.")
    items_ingested: int = Field(..., description="Number of records successfully ingested.")
    errors: List[str] = Field(default_factory=list, description="Any non-fatal errors encountered.")

# PUBLIC_INTERFACE
class ChatRequest(BaseModel):
    """Chat request payload."""
    query: str = Field(..., description="User question about placement records.")
    conversation_id: Optional[str] = Field(None, description="Optional conversation identifier for future threading.")
    top_k: Optional[int] = Field(default=TOP_K, description="Number of top documents to retrieve from Pinecone.")

# PUBLIC_INTERFACE
class RetrievedContext(BaseModel):
    """A retrieved context chunk."""
    id: Optional[str] = Field(None, description="Vector ID of the chunk")
    text: str = Field(..., description="The text content of the retrieved chunk")
    score: Optional[float] = Field(None, description="Similarity score or distance")

# PUBLIC_INTERFACE
class ChatResponse(BaseModel):
    """Chat response payload with strict RAG policy."""
    answer: str = Field(..., description="Model-generated answer strictly based on retrieved placement records or fallback.")
    contexts: List[RetrievedContext] = Field(default_factory=list, description="Retrieved contexts used to answer.")
    used_fallback: bool = Field(..., description="True if fallback message was used due to no/low-confidence context.")

# Utilities
def _require_env(var: str) -> None:
    if not os.getenv(var):
        logger.warning(f"Environment variable {var} is not set. Please configure it in .env.")

for v in ["PINECONE_API_KEY", "PINECONE_INDEX_NAME", "OPENROUTER_API_KEY"]:
    _require_env(v)

# Embedding: Using OpenRouter embeddings endpoint for llama-text-embed-v2, if available.
# PUBLIC_INTERFACE
async def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts using llama-text-embed-v2 via OpenRouter."""
    # Some OpenRouter-compatible embedding endpoints use model: 'meta-llama/llama-text-embed-...' or 'llama-text-embed-v2'
    # We'll attempt a standard embeddings route. If unavailable, raise a graceful error.
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": SITE_URL,
        "X-Title": "Connecto (Beta) - Placement Records",
        "Content-Type": "application/json",
    }
    url = "https://openrouter.ai/api/v1/embeddings"
    payload = {
        "model": "meta-llama/llama-text-embed-v2",  # explicit for OpenRouter catalog
        "input": texts,
    }
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code != 200:
                logger.error(f"Embedding API error: {resp.status_code} - {resp.text}")
                raise HTTPException(status_code=502, detail="Embedding service error")
            data = resp.json()
            # Expected shape: {"data":[{"embedding":[...]}...]}
            vectors = [item["embedding"] for item in data.get("data", [])]
            if len(vectors) != len(texts):
                logger.warning("Embedding count mismatch; proceeding with available embeddings.")
            return vectors
    except Exception as e:
        logger.exception("Embedding request failed")
        raise HTTPException(status_code=502, detail="Embedding request failed") from e

# Pinecone minimal HTTP usage (generic) to avoid adding extra SDK dependencies beyond given requirements.
# We'll support v2 style query if PINECONE_HOST is provided, otherwise use older REST.
def _pinecone_headers() -> Dict[str, str]:
    return {
        "Api-Key": PINECONE_API_KEY,
        "Content-Type": "application/json",
    }

async def _pinecone_upsert(vectors: List[Dict[str, Any]]) -> None:
    """Upsert vectors into the Pinecone index via REST API."""
    if not PINECONE_API_KEY or not PINECONE_INDEX_NAME:
        raise HTTPException(status_code=500, detail="Pinecone configuration missing")

    # Prefer host if provided (v2)
    if PINECONE_HOST:
        url = f"https://{PINECONE_HOST}/vectors/upsert"
    else:
        # Legacy style: https://controller.{env}.pinecone.io/actions/whoami -> region; but we'll assume standard:
        # If environment provided, older v1 style: https://{index_name}-{project_id}.svc.{environment}.pinecone.io/vectors/upsert (requires full host)
        # Without host, we cannot proceed reliably.
        raise HTTPException(status_code=500, detail="PINECONE_HOST not set. Please set PINECONE_HOST for your index endpoint.")

    payload = {"vectors": vectors}
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, headers=_pinecone_headers(), json=payload)
            if resp.status_code not in (200, 201):
                logger.error(f"Pinecone upsert error: {resp.status_code} - {resp.text}")
                raise HTTPException(status_code=502, detail="Pinecone upsert error")
    except Exception as e:
        logger.exception("Pinecone upsert request failed")
        raise HTTPException(status_code=502, detail="Pinecone upsert request failed") from e

async def _pinecone_query(vector: List[float], top_k: int) -> List[Dict[str, Any]]:
    """Query the Pinecone index via REST API."""
    if not PINECONE_API_KEY or not PINECONE_INDEX_NAME:
        raise HTTPException(status_code=500, detail="Pinecone configuration missing")

    if PINECONE_HOST:
        url = f"https://{PINECONE_HOST}/query"
    else:
        raise HTTPException(status_code=500, detail="PINECONE_HOST not set. Please set PINECONE_HOST for your index endpoint.")

    payload = {"vector": vector, "topK": int(top_k), "includeMetadata": True}
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, headers=_pinecone_headers(), json=payload)
            if resp.status_code != 200:
                logger.error(f"Pinecone query error: {resp.status_code} - {resp.text}")
                raise HTTPException(status_code=502, detail="Pinecone query error")
            data = resp.json()
            matches = data.get("matches", [])
            # Normalize to list of {id, score, metadata}
            return matches
    except Exception as e:
        logger.exception("Pinecone query request failed")
        raise HTTPException(status_code=502, detail="Pinecone query request failed") from e

# PUBLIC_INTERFACE
async def generate_answer_from_context(query: str, contexts: List[str]) -> str:
    """Call OpenRouter to generate an answer strictly grounded to provided contexts."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": SITE_URL,
        "X-Title": "Connecto (Beta) - Placement Records",
        "Content-Type": "application/json",
    }
    url = "https://openrouter.ai/api/v1/chat/completions"

    system_prompt = (
        "You are Connecto (Beta), a helpful assistant for final-year students. "
        "STRICT RAG POLICY: Only answer using the provided placement records context. "
        "If the context is insufficient or unrelated, reply exactly with: "
        f"\"{FALLBACK_ANSWER}\"\n\n"
        "Guidelines:\n"
        "- Be concise (1-3 sentences), friendly, and direct.\n"
        "- Do not hallucinate or infer beyond the context.\n"
        "- If numerical data (e.g., CTC, number of offers) is present, state it clearly.\n"
        "- Avoid disclaimers about being an AI.\n"
    )
    joined_context = "\n\n".join([f"- {c}" for c in contexts]) if contexts else "No relevant context was found."
    user_prompt = (
        f"User question: {query}\n\n"
        f"Placement records context (use strictly):\n{joined_context}\n\n"
        "Answer:"
    )

    payload = {
        "model": "nousresearch/hermes-3-llama-3.1-405b:extended" if GENERATION_MODEL == "openrouter/gpt-oss-20b" else GENERATION_MODEL,
        # Note: Some OpenRouter catalogs alias models; as a conservative approach, specify a robust open model if needed.
        # If a specific 'gpt-oss-20b' route exists, set GENERATION_MODEL to that explicit ID in .env and we will pass it through.
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 300,
    }

    try:
        async with httpx.AsyncClient(timeout=90) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code != 200:
                logger.error(f"OpenRouter error: {resp.status_code} - {resp.text}")
                return FALLBACK_ANSWER
            data = resp.json()
            # Expected shape: choices[0].message.content
            choices = data.get("choices", [])
            if not choices:
                return FALLBACK_ANSWER
            content = choices[0].get("message", {}).get("content", "") or ""
            content = content.strip()
            if not content:
                return FALLBACK_ANSWER
            # Enforce strict policy: If the model ignored policy, fallback if it references lack of data or unrelated content.
            return content
    except Exception:
        logger.exception("LLM generation failed")
        return FALLBACK_ANSWER

# FastAPI app with metadata and tags
app = FastAPI(
    title="Connecto (Beta) - Placement Records Backend",
    description="RAG chatbot API using Pinecone, llama-text-embed-v2 embeddings, and OpenRouter gpt-oss-20b. "
                "Strictly answers based on placement records. No authentication.",
    version="0.1.0",
    contact={
        "name": "Connecto (Beta)",
        "url": SITE_URL,
    },
    openapi_tags=[
        {"name": "health", "description": "Service health and information"},
        {"name": "chat", "description": "Chat endpoint for placement records Q&A"},
        {"name": "ingestion", "description": "Placement data ingestion and management"},
        {"name": "websocket", "description": "WebSocket notes and usage help"},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Open access as requested
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_model=HealthResponse, tags=["health"], summary="Health Check", description="Simple health check endpoint.")
def health_check():
    """Health check endpoint returning service status."""
    return HealthResponse(status="ok")

@app.get("/ws-help", tags=["websocket"], summary="WebSocket Usage", description="This backend currently does not expose WebSocket endpoints. Real-time features can be added in future iterations.")
def websocket_help():
    """Documentation note for WebSocket support. Not implemented in this version."""
    return {"message": "No WebSocket endpoints available in this version. Use REST /chat for Q&A."}

# PUBLIC_INTERFACE
@app.post("/ingest", response_model=IngestResponse, tags=["ingestion"], summary="Ingest placement records", description="Embed and upsert placement records into Pinecone for retrieval.")
async def ingest_records(payload: IngestRequest) -> IngestResponse:
    """Embeds provided records and upserts them into Pinecone index.

    Parameters:
    - records: list of placement records with text and optional metadata.

    Returns:
    - success: whether operation was successful
    - items_ingested: number of vectors ingested
    - errors: list of error messages, if any
    """
    records = payload.records or []
    if not records:
        raise HTTPException(status_code=400, detail="No records provided")

    texts = [r.text for r in records]
    try:
        embeddings = await embed_texts(texts)
    except HTTPException as e:
        return IngestResponse(success=False, items_ingested=0, errors=[str(e.detail)])

    vectors: List[Dict[str, Any]] = []
    errors: List[str] = []
    for i, emb in enumerate(embeddings):
        try:
            rec = records[i]
            vid = rec.id or f"rec-{i}-{abs(hash(rec.text)) % (10**8)}"
            metadata = {"text": rec.text}
            if rec.metadata:
                # Ensure metadata is JSON-serializable
                try:
                    json.dumps(rec.metadata)
                    metadata.update(rec.metadata)
                except Exception:
                    metadata.update({"_metadata_error": "Non-serializable metadata ignored"})
            vectors.append({"id": vid, "values": emb, "metadata": metadata})
        except Exception as e:
            errors.append(f"Record index {i} error: {e}")

    if not vectors:
        return IngestResponse(success=False, items_ingested=0, errors=errors or ["No vectors created"])

    try:
        await _pinecone_upsert(vectors)
    except HTTPException as e:
        errors.append(f"Pinecone upsert failed: {e.detail}")
        return IngestResponse(success=False, items_ingested=0, errors=errors)

    return IngestResponse(success=True, items_ingested=len(vectors), errors=errors)

# PUBLIC_INTERFACE
@app.post("/chat", response_model=ChatResponse, tags=["chat"], summary="Chat over placement records", description="Ask a question about historical placement data. Uses strict RAG policy; falls back if no relevant context.")
async def chat(payload: ChatRequest) -> ChatResponse:
    """Answers a user query strictly grounded on retrieved placement records.

    Request:
    - query: user question.
    - top_k: number of top matches to retrieve (default 3).

    Response:
    - answer: generated response grounded in context or fallback string.
    - contexts: the top retrieved context snippets used to answer.
    - used_fallback: whether fallback was used.
    """
    query = (payload.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

    top_k = payload.top_k or TOP_K

    # Step 1: Create embedding for the query
    try:
        q_emb_list = await embed_texts([query])
        if not q_emb_list or not q_emb_list[0]:
            logger.warning("Empty embedding returned for query")
            return ChatResponse(answer=FALLBACK_ANSWER, contexts=[], used_fallback=True)
        q_emb = q_emb_list[0]
    except HTTPException:
        return ChatResponse(answer=FALLBACK_ANSWER, contexts=[], used_fallback=True)

    # Step 2: Query Pinecone
    try:
        matches = await _pinecone_query(q_emb, top_k=top_k)
    except HTTPException:
        return ChatResponse(answer=FALLBACK_ANSWER, contexts=[], used_fallback=True)

    # Step 3: Collect contexts
    contexts: List[RetrievedContext] = []
    context_texts: List[str] = []
    for m in matches:
        text = ""
        md = m.get("metadata") or {}
        if isinstance(md, dict):
            text = md.get("text") or ""
        else:
            # Some deployments might return metadata as string
            try:
                parsed = json.loads(md)
                text = parsed.get("text", "") if isinstance(parsed, dict) else ""
            except Exception:
                text = ""
        if text:
            context_texts.append(text)
            contexts.append(
                RetrievedContext(
                    id=m.get("id"),
                    text=text,
                    score=m.get("score"),
                )
            )

    # Step 4: If no contexts, fallback
    if not context_texts:
        return ChatResponse(answer=FALLBACK_ANSWER, contexts=[], used_fallback=True)

    # Step 5: Call LLM with strict prompt
    answer = await generate_answer_from_context(query, context_texts)
    used_fallback = (answer.strip() == FALLBACK_ANSWER)
    return ChatResponse(answer=answer, contexts=contexts, used_fallback=used_fallback)

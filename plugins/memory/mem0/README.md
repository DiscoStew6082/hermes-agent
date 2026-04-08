# Mem0 Memory Provider

Two modes supported:

1. **Platform Mode** — Uses Mem0 Platform API with server-side LLM extraction (requires API key)
2. **Local/OSS Mode** — Self-hosted using LM Studio or Ollama for local inference (no API key needed)

## Requirements

### For Platform Mode
- `pip install mem0ai`
- Mem0 API key from [app.mem0.ai](https://app.mem0.ai)

### For Local/OSS Mode  
- `pip install mem0ai faiss-cpu`
- LM Studio or Ollama running locally (e.g., `http://localhost:1234/v1`)
- Embedding model loaded in LM Studio
- LLM model loaded in LM Studio

## Setup

```bash
hermes memory setup    # select "mem0" for platform mode, or configure local mode manually
```

Or manually:
```bash
hermes config set memory.provider mem0
echo "MEM0_API_KEY=*** >> ~/.hermes/.env
```

### Local/OSS Mode Configuration

Create `$HERMES_HOME/mem0.json` with:

```json
{
  "mode": "local",
  "llm_provider": "lmstudio",
  "llm_model": "google/gemma-4-26b-a4b",
  "llm_base_url": "http://localhost:1234/v1",
  "embedder_provider": "lmstudio", 
  "embedder_model": "text-embedding-kalm-embedding-gemma3-12b-2511-i1",
  "embedder_dims": 3840,
  "vector_store_provider": "faiss",
  "vector_store_path": "/tmp/mem0_hermes"
}
```

Then set the memory provider:
```bash
hermes config set memory.provider mem0
```

## Config

Config file: `$HERMES_HOME/mem0.json`

| Key | Default | Description |
|-----|---------|-------------|
| `mode` | `platform` | Mode selection: "platform" or "local" |
| `user_id` | `hermes-user` | User identifier on Mem0 (platform mode) |
| `agent_id` | `hermes` | Agent identifier (platform mode) |
| `rerank` | `true` | Enable reranking for recall |

### Local Mode Keys

| Key | Default | Description |
|-----|---------|-------------|
| `llm_provider` | `lmstudio` | LLM provider: lmstudio, ollama, openai |
| `llm_model` | `google/gemma-4-26b-a4b` | Model for fact extraction |
| `llm_base_url` | `http://localhost:1234/v1` | API base URL |
| `embedder_provider` | `lmstudio` | Embedding provider |
| `embedder_model` | `text-embedding-kalm-embedding-gemma3-12b-2511-i1` | Embedding model |
| `embedder_dims` | `3840` | Embedding dimensions |
| `vector_store_provider` | `faiss` | Vector store: faiss, qdrant, chroma |
| `vector_store_path` | `/tmp/mem0_hermes` | Storage path for faiss/chroma |

## Tools

| Tool | Description |
|------|-------------|
| `mem0_profile` | All stored memories about the user |
| `mem0_search` | Semantic search with optional reranking |
| `mem0_conclude` | Store a fact verbatim (no LLM extraction) |

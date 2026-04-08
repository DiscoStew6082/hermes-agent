# Mem0 Local Mode Setup - From Official Docs

Source: https://docs.mem0.ai/components/llms/models/lmstudio

## Prerequisites

1. **LM Studio** running with server enabled at `http://localhost:1234/v1`
2. Load these models in LM Studio:
   - Embedding model for vector embeddings
   - LLM model for fact extraction

## Step 1: Install Dependencies

```bash
source ~/.hermes/hermes-agent/venv/bin/activate
uv pip install faiss-cpu mem0ai
```

## Step 2: Create Config File

According to official docs, the config format is:

```json
{
  "mode": "local",
  "llm_provider": "lmstudio",
  "embedder_provider": "lmstudio", 
  "vector_store_provider": "faiss"
}
```

Optional parameters you can add:
- `llm_model` - LLM model name (default varies by mem0 version)
- `llm_base_url` - defaults to http://localhost:1234/v1
- `embedder_model` - embedding model name  
- `embedder_dims` - embedding dimensions (default 1536)

## Step 3: Enable Provider

```bash
hermes config set memory.provider mem0
```

## Step 4: Test

Run hermes and verify LM Studio is receiving requests at http://localhost:1234/v1
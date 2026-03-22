# VaultTools — Personal Knowledge Base RAG System

Local vector database (ChromaDB) + RAG system for searching personal documents stored in `~/Public`.

## Architecture

```
~/Public/                     ← Source documents (PDF, XLSX, CSV, PPTX, MD, TXT)
    ├── 00_Career/
    ├── 10_Finance/
    ├── 20_Medical/
    ├── 30_Travel/
    ├── 40_Resources/
    ├── 60_Archive/
    └── 永住/

~/Documents/VaultTools/       ← This repo
    ├── ingest.py             ← Extract → Markdown → Chunk → Embed → Store
    ├── query.py              ← Search → Filter → LLM Answer
    ├── auto_sync.py          ← Detect new/modified/deleted → Sync
    ├── browse_db.py          ← Browse the ChromaDB database
    ├── inspect_db.py         ← Inspect database statistics
    └── .chromadb/            ← ChromaDB vector database (gitignored)
```

## Requirements

- **Python 3.9+**
- **Ollama** running locally (`http://localhost:11434`)
  - Embedding model: `nomic-embed-text`
  - Chat model: `qwen3:8b`
- Global Python packages: **chromadb**, **pypdf**, **openpyxl**, **python-pptx**

## Setup

```bash
# Install dependencies globally
pip3 install chromadb pypdf openpyxl python-pptx
```

## Usage

### 1. Ingest documents

Scan `~/Public`, convert to Markdown, chunk, embed, and store in ChromaDB:

```bash
python3 ingest.py              # Full vault
python3 ingest.py 00_Career    # Specific subfolder only
```

### 2. Query the knowledge base

```bash
python3 query.py "最近2年的保險費用"
python3 query.py "2024年の旅行プラン"
python3 query.py "AWS証明書は何を持っていますか？"
```

### 3. Auto-sync (detect changes)

Detects new, modified, or deleted files and syncs to ChromaDB:

```bash
python3 auto_sync.py
```

### 4. Scheduled auto-sync (launchd)

A plist is installed at `~/Library/LaunchAgents/com.vaulttools.autosync.plist` for twice-daily sync:

| Time  | Action |
|-------|--------|
| 08:00 | Auto-sync |
| 20:00 | Auto-sync |

```bash
# Load the schedule
launchctl load ~/Library/LaunchAgents/com.vaulttools.autosync.plist

# Unload
launchctl unload ~/Library/LaunchAgents/com.vaulttools.autosync.plist

# Check status
launchctl list | grep vaulttools
```

## Configuration

Key parameters in each script:

### ingest.py

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CHUNK_SIZE` | 1500 | Characters per chunk |
| `CHUNK_OVERLAP` | 300 | Overlap between chunks |
| `VAULT_PATH` | `~/Public` | Source documents path |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model |

### query.py

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TOP_K` | 10 | Candidates to retrieve before filtering |
| `MIN_SIMILARITY` | 0.55 | Minimum cosine similarity threshold |
| `MAX_CHUNKS_PER_FILE` | 2 | Max chunks per unique file |
| `MAX_RESULTS` | 5 | Final results after dedup |
| `OLLAMA_CHAT_MODEL` | `qwen3:8b` | LLM for answer generation |

## Pipeline

```
Document → Extract Text → Convert to Markdown → Chunk (1500 chars)
    → Embed (nomic-embed-text) → Store in ChromaDB

Query → Embed → Vector Search (top 10)
    → Filter (similarity ≥ 0.55) → Dedup → LLM Answer (qwen3:8b)
```

## OpenClaw Integration

This system is integrated as an OpenClaw skill (`vault-search`):

```
/rag <query>    — Search the knowledge base
```

Skill config location: `~/.openclaw/skills/vault-search/`

## Rebuild Database

To fully rebuild from scratch:

```bash
rm -rf .chromadb
source .venv/bin/activate
python3 ingest.py
```

## Logs

| Log File | Location | Description |
|----------|----------|-------------|
| LLM Dialogue | `logs/llm_dialogue_log.txt` | Query + Context + Answer log |
| Auto-sync | `logs/auto_sync.log` | Sync operations log |
| Launchd stdout | `logs/launchd_sync_stdout.log` | Scheduled sync output |
| Launchd stderr | `logs/launchd_sync_stderr.log` | Scheduled sync errors |

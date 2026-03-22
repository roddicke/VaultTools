#!/usr/bin/env python3
"""
query.py - Search the Public vault vector database using natural language.

Usage:
    source /Users/chaomingou/Documents/VaultTools/.venv/bin/activate
    python /Users/chaomingou/Documents/VaultTools/query.py "我去年去日本花了多少錢？"
"""

import sys
import os
import json
import urllib.request
import chromadb
import datetime
from pathlib import Path
from typing import Optional, List
from collections import defaultdict

# ─── Configuration ───────────────────────────────────────────────────────────

CHROMA_PATH = Path("/Users/chaomingou/Documents/VaultTools/.chromadb")
COLLECTION_NAME = "public_vault"
OLLAMA_EMBED_MODEL = "nomic-embed-text"
OLLAMA_CHAT_MODEL = "qwen3:8b"
TOP_K = 10  # Retrieve more candidates, then filter
MIN_SIMILARITY = 0.55  # Minimum similarity threshold
MAX_CHUNKS_PER_FILE = 2  # Max chunks per unique file
MAX_RESULTS = 5  # Final max results after dedup/filtering


# ─── Embedding ───────────────────────────────────────────────────────────────

def get_embedding(text: str) -> Optional[List[float]]:
    """Get embedding vector from Ollama."""
    prefixed_text = f"search_query: {text}"
    try:
        data = json.dumps({"model": OLLAMA_EMBED_MODEL, "input": prefixed_text}).encode()
        req = urllib.request.Request(
            "http://localhost:11434/api/embed",
            data=data,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
            embeddings = result.get("embeddings")
            if embeddings and len(embeddings) > 0:
                return embeddings[0]
        return None
    except Exception as e:
        print(f"⚠ Embedding error: {e}")
        return None


# ─── Result Filtering ────────────────────────────────────────────────────────

def filter_and_dedup(documents, metadatas, distances):
    """Filter by similarity threshold, deduplicate by filename, limit per file."""
    # Build candidate list with similarity scores
    candidates = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        similarity = 1 - dist
        if similarity < MIN_SIMILARITY:
            continue
        candidates.append({
            "doc": doc,
            "meta": meta,
            "similarity": similarity,
            "filename": meta.get("filename", "Unknown"),
            "category": meta.get("category", "Unknown"),
        })

    # Sort by similarity descending
    candidates.sort(key=lambda x: x["similarity"], reverse=True)

    # Deduplicate: for the same filename, keep only the best chunks
    # (same content from different categories counts as duplicate)
    file_chunks = defaultdict(list)
    seen_docs = set()

    for c in candidates:
        # Use first 200 chars of doc content as dedup key
        doc_key = c["doc"][:200]
        if doc_key in seen_docs:
            continue
        seen_docs.add(doc_key)

        fname = c["filename"]
        if len(file_chunks[fname]) < MAX_CHUNKS_PER_FILE:
            file_chunks[fname].append(c)

    # Flatten and re-sort, then limit
    results = []
    for chunks in file_chunks.values():
        results.extend(chunks)
    results.sort(key=lambda x: x["similarity"], reverse=True)

    return results[:MAX_RESULTS]


# ─── LLM Answer ─────────────────────────────────────────────────────────────

def ask_llm(question: str, context: str) -> str:
    """Use Ollama to generate an answer based on retrieved context."""
    prompt = f"""あなたは個人知識庫（Personal Knowledge Base）の検索アシスタントです。
以下の検索結果（Context）に基づいて、ユーザーの質問に回答してください。

## 検索結果 (Context):
{context}

## ユーザーの質問:
{question}

## 回答ルール:
1. 検索結果に少しでも関連する情報があれば、必ずそれを元に回答してください。
2. 複数の検索結果から情報を総合・整理して、わかりやすく回答してください。
3. 具体的な日付、金額、名称がある場合は必ず含めてください。
4. 検索結果のカテゴリ（category）も参考にして回答の文脈を判断してください。
5. 検索結果が質問に直接的に回答していなくても、関連する情報があればそれを提示してください。
   例：「保険費用」を聞かれた時、具体的な支払額がなくても、保険の契約内容やプランの情報があれば整理して提示。
6. 「見つかりませんでした」と回答するのは、検索結果が質問と全く無関係な場合のみにしてください。
7. 言語はユーザーの質問に合わせて、日本語または中国語で答えてください。

## 回答:"""

    try:
        data = json.dumps({
            "model": OLLAMA_CHAT_MODEL,
            "prompt": prompt,
            "stream": False
        }).encode()
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=data,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode())
            return result.get("response", "回答を生成できませんでした。")
    except Exception as e:
        return f"⚠ LLM error: {e}"


# ─── Main Query ──────────────────────────────────────────────────────────────

def query(question: str, show_sources: bool = True):
    """Search the vector database and answer the question."""
    print(f"\n🔍 Question: {question}\n")

    # Connect to ChromaDB
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_collection(name=COLLECTION_NAME)

    # Get embedding for the question
    q_embedding = get_embedding(question)
    if q_embedding is None:
        print("❌ Failed to generate embedding for the question.")
        return

    # Search with more candidates
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"]
    )

    if not results["documents"][0]:
        print("❌ No relevant documents found.")
        return

    # Filter, deduplicate, and limit results
    filtered = filter_and_dedup(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )

    if not filtered:
        print("❌ No results above similarity threshold.")
        return

    # Build context from filtered results
    context_parts = []
    sources = []
    for r in filtered:
        sources.append(f"[{r['category']}] {r['filename']} (similarity: {r['similarity']:.2f})")
        context_parts.append(
            f"--- Source: {r['filename']} (category: {r['category']}, similarity: {r['similarity']:.2f}) ---\n{r['doc']}\n"
        )

    context = "\n".join(context_parts)

    # Show sources
    if show_sources:
        print("📚 Sources found:")
        for s in sources:
            print(f"   • {s}")
        print()

    # Ask LLM
    print("💬 Answer:")
    answer = ask_llm(question, context)
    print(answer)
    print()

    # Log the query, context, and answer
    log_entry = f"[{datetime.datetime.now().isoformat()}]\nQuestion: {question}\nContext:\n{context}\nAnswer:\n{answer}\n{'-'*40}\n"
    with open('/Users/chaomingou/Documents/VaultTools/logs/llm_dialogue_log.txt', 'a', encoding='utf-8') as f:
        f.write(log_entry)


# ─── Entry Point ─────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python query.py \"your question here\"")
        print()
        print("Examples:")
        print('  python query.py "我去年去日本花了多少錢？"')
        print('  python query.py "AWS証明書は何を持っていますか？"')
        print('  python query.py "醫療費用有多少？"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    query(question)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
query.py - Search the Public vault vector database using natural language.

Usage:
    source /Users/chaomingou/Documents/VaultTools/.venv/bin/activate
    python /Users/chaomingou/Documents/VaultTools/query.py "我去年去日本花了多少錢？"
"""

import sys
import json
import urllib.request
import chromadb
from pathlib import Path
from typing import Optional, List

# ─── Configuration ───────────────────────────────────────────────────────────

CHROMA_PATH = Path("/Users/chaomingou/Documents/VaultTools/.chromadb")
COLLECTION_NAME = "public_vault"
OLLAMA_EMBED_MODEL = "nomic-embed-text"
OLLAMA_CHAT_MODEL = "qwen3:8b"
TOP_K = 5  # Number of results to retrieve


# ─── Embedding ───────────────────────────────────────────────────────────────

def get_embedding(text: str) -> Optional[List[float]]:
    """Get embedding vector from Ollama."""
    # Prefix for nomic-embed-text
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


# ─── LLM Answer ─────────────────────────────────────────────────────────────

def ask_llm(question: str, context: str) -> str:
    """Use Ollama to generate an answer based on retrieved context."""
    prompt = f"""あなたは知識庫（Local Documents）の検索アシスタントです。
以下の検索結果（Context）に基づいて、ユーザーの質問に回答してください。

## 検索結果 (Context):
{context}

## ユーザーの質問:
{question}

## 指示:
1. 検索結果の中に少しでも関連する情報があれば、それを元に回答してください。
2. もし具体的な日付や金額があれば、必ず含めてください。
3. 答えが全く見つからない場合のみ、「この情報は知識庫に見つかりませんでした」と答えてください。
4. 言語はユーザーの質問に合わせて、日本語または中国語で答えてください。

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
    
    # Search
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"]
    )

    if not results["documents"][0]:
        print("❌ No relevant documents found.")
        return
    
    # Build context from results
    context_parts = []
    sources = set()
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        similarity = 1 - dist  # cosine distance to similarity
        source = meta.get("filename", "Unknown")
        category = meta.get("category", "Unknown")
        sources.add(f"[{category}] {source}")
        context_parts.append(f"--- Source: {source} (category: {category}, similarity: {similarity:.2f}) ---\n{doc}\n")
    
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

#!/usr/bin/env python3
"""
auto_sync.py - Automatically detect new, modified, or deleted files
in the Public vault and sync them to the ChromaDB vector database.

Usage:
    source Path(__file__).resolve().parent/.venv/bin/activate
    python Path(__file__).resolve().parent/auto_sync.py

Designed to be run via launchd (twice daily).
"""

import sys
import os
import json
import hashlib
import datetime
import chromadb
from pathlib import Path
from typing import Optional, List

# Import shared functions from ingest.py
import sys
sys.path.insert(0, str(Path(__file__).parent))
from ingest import (
    VAULT_PATH, CHROMA_PATH, COLLECTION_NAME, OLLAMA_EMBED_MODEL,
    SUPPORTED_EXTENSIONS, SKIP_DIRS, CHUNK_SIZE, CHUNK_OVERLAP,
    extract_text, convert_to_markdown, chunk_text, get_embedding,
    discover_files, file_hash, detect_category
)

BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = BASE_DIR / "logs" / "auto_sync.log"


def log(msg: str):
    """Log a timestamped message."""
    ts = datetime.datetime.now().isoformat()
    line = f"[{ts}] {msg}"
    print(line)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def get_indexed_files(collection) -> dict:
    """Get a mapping of source path -> file_hash from the collection metadata."""
    indexed = {}
    try:
        # Get all metadatas from the collection
        all_data = collection.get(include=["metadatas"])
        if all_data and all_data["metadatas"]:
            for meta in all_data["metadatas"]:
                source = meta.get("source", "")
                fhash = meta.get("file_hash", "")
                if source and fhash:
                    indexed[source] = fhash
    except Exception as e:
        log(f"⚠ Error reading indexed files: {e}")
    return indexed


def remove_file_chunks(collection, source_path: str):
    """Remove all chunks for a given source file from the collection."""
    try:
        all_data = collection.get(include=["metadatas"])
        ids_to_delete = []
        for i, meta in enumerate(all_data["metadatas"]):
            if meta.get("source") == source_path:
                ids_to_delete.append(all_data["ids"][i])
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
            log(f"  🗑 Removed {len(ids_to_delete)} chunks for {Path(source_path).name}")
    except Exception as e:
        log(f"  ⚠ Error removing chunks: {e}")


def ingest_file(collection, filepath: Path):
    """Ingest a single file into the collection."""
    category = detect_category(filepath)
    fhash = file_hash(filepath)

    text = extract_text(filepath)
    if not text.strip():
        log(f"  ⏭ No text extracted from {filepath.name}")
        return 0

    md_text = convert_to_markdown(text, filepath)
    chunks = chunk_text(md_text)
    if not chunks:
        log(f"  ⏭ No chunks from {filepath.name}")
        return 0

    stored = 0
    for j, chunk in enumerate(chunks):
        chunk_id = f"{fhash}_{j}"
        embedding = get_embedding(chunk)
        if embedding is None:
            continue
        collection.add(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{
                "source": str(filepath),
                "filename": filepath.name,
                "category": category,
                "chunk_index": j,
                "total_chunks": len(chunks),
                "file_hash": fhash,
            }]
        )
        stored += 1

    return stored


def main():
    log("=" * 50)
    log("🔄 Auto-sync started")

    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # Discover current files
    current_files = discover_files(VAULT_PATH)
    current_map = {}  # source_path -> file_hash
    for f in current_files:
        current_map[str(f)] = file_hash(f)

    # Get indexed files
    indexed_map = get_indexed_files(collection)

    # Find new, modified, and deleted files
    new_files = []
    modified_files = []
    deleted_sources = []

    for source, fhash in current_map.items():
        if source not in indexed_map:
            new_files.append(Path(source))
        elif indexed_map[source] != fhash:
            modified_files.append(Path(source))

    indexed_sources = set(indexed_map.keys())
    current_sources = set(current_map.keys())
    for source in indexed_sources - current_sources:
        deleted_sources.append(source)

    log(f"📊 Status: {len(current_files)} total files, "
        f"{len(new_files)} new, {len(modified_files)} modified, "
        f"{len(deleted_sources)} deleted")

    total_chunks = 0

    # Handle deleted files
    for source in deleted_sources:
        log(f"❌ Deleted: {Path(source).name}")
        remove_file_chunks(collection, str(source))

    # Handle modified files (remove old, re-ingest)
    for filepath in modified_files:
        log(f"🔄 Modified: {filepath.name}")
        remove_file_chunks(collection, str(filepath))
        chunks = ingest_file(collection, filepath)
        total_chunks += chunks
        log(f"  ✅ Re-ingested {chunks} chunks")

    # Handle new files
    for filepath in new_files:
        log(f"➕ New: {filepath.name}")
        chunks = ingest_file(collection, filepath)
        total_chunks += chunks
        log(f"  ✅ Ingested {chunks} chunks")

    log(f"✅ Sync complete! Chunks added/updated: {total_chunks}")
    log("=" * 50)


if __name__ == "__main__":
    main()

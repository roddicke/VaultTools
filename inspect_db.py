#!/usr/bin/env python3
import chromadb
from pathlib import Path
import sys

CHROMA_PATH = Path("/Users/chaomingou/Developer/VaultTools/.chromadb")
COLLECTION_NAME = "public_vault"

def inspect():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        print(f"❌ Error: Collection '{COLLECTION_NAME}' not found. Have you run ingest.py?")
        return

    count = collection.count()
    print(f"📊 Total chunks in DB: {count}")

    # Get distinct sources
    results = collection.get(include=["metadatas"])
    sources = set()
    categories = set()
    for meta in results["metadatas"]:
        sources.add(meta.get("filename", "Unknown"))
        categories.add(meta.get("category", "Unknown"))

    print(f"📂 Total unique files: {len(sources)}")
    print(f"🏷️ Categories: {', '.join(sorted(categories))}")
    print("\n📄 Top 10 Files (Sample):")
    for s in sorted(list(sources))[:10]:
        print(f"   • {s}")

    if len(sys.argv) > 1 and sys.argv[1] == "--list-all":
        print("\n📜 Full File List:")
        for s in sorted(list(sources)):
            print(f"   • {s}")

if __name__ == "__main__":
    inspect()

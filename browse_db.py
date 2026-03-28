#!/usr/bin/env python3
import chromadb
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
CHROMA_PATH = BASE_DIR / ".chromadb"
COLLECTION_NAME = "public_vault"

def main():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        print(f"❌ Error: Collection '{COLLECTION_NAME}' not found.")
        return

    while True:
        print("\n--- ChromaDB Browser ---")
        print("1. List all unique files")
        print("2. Show statistics by category")
        print("3. View content of a specific file")
        print("4. List all chunks (Paginated)")
        print("5. Find similar chunks (by ID)")
        print("q. Quit")
        
        choice = input("\nSelect an option: ").strip().lower()
        
        if choice == '1':
            results = collection.get(include=["metadatas"])
            filenames = sorted(list(set(m.get("filename", "Unknown") for m in results["metadatas"])))
            print(f"\n📂 Files in DB ({len(filenames)} total):")
            for i, f in enumerate(filenames, 1):
                print(f"  {i:3}. {f}")
                
        elif choice == '2':
            results = collection.get(include=["metadatas"])
            categories = {}
            for m in results["metadatas"]:
                cat = m.get("category", "Unknown")
                categories[cat] = categories.get(cat, 0) + 1
            print("\n🏷️ Statistics by Category:")
            for cat, count in sorted(categories.items()):
                print(f"  • {cat:15}: {count} chunks")
                
        elif choice == '3':
            filename = input("Enter filename (or part of it): ").strip()
            if not filename: continue
            
            # Find matching files
            results = collection.get(include=["metadatas"])
            all_files = sorted(list(set(m.get("filename", "Unknown") for m in results["metadatas"])))
            matches = [f for f in all_files if filename.lower() in f.lower()]
            
            if not matches:
                print("❌ No matching files found.")
                continue
            
            if len(matches) > 1:
                print("\nMultiple matches found:")
                for i, f in enumerate(matches, 1):
                    print(f"  {i}. {f}")
                idx = input("Select file index (or 'c' to cancel): ").strip()
                if idx.lower() == 'c': continue
                try:
                    target_file = matches[int(idx)-1]
                except (ValueError, IndexError):
                    print("Invalid selection.")
                    continue
            else:
                target_file = matches[0]
            
            # Get chunks for this file
            results = collection.get(
                where={"filename": target_file},
                include=["documents", "metadatas"]
            )
            
            print(f"\n📄 Content for: {target_file}")
            # Sort by chunk_index
            chunks = sorted(zip(results["documents"], results["metadatas"]), key=lambda x: x[1].get("chunk_index", 0))
            
            for i, (doc, meta) in enumerate(chunks):
                print(f"\n--- Chunk {meta.get('chunk_index', i)} ---")
                print(doc)
                if i < len(chunks) - 1:
                    input("\nPress Enter for next chunk...")
            print("\n--- End of file ---")

        elif choice == '4':
            limit = 10
            offset = 0
            while True:
                results = collection.get(
                    limit=limit,
                    offset=offset,
                    include=["documents", "metadatas"]
                )
                
                if not results["ids"]:
                    print("\nNo more chunks.")
                    break
                
                print(f"\n--- Showing chunks {offset+1} to {offset+len(results['ids'])} ---")
                for i in range(len(results["ids"])):
                    doc = results["documents"][i]
                    meta = results["metadatas"][i]
                    print(f"\n[ID: {results['ids'][i]}] File: {meta.get('filename')} (Category: {meta.get('category')})")
                    print(f"Content snippet: {doc[:200]}...")
                
                cmd = input("\n[n] Next page, [p] Previous page, [q] Return to menu: ").strip().lower()
                if cmd == 'n':
                    offset += limit
                elif cmd == 'p':
                    offset = max(0, offset - limit)
                elif cmd == 'q':
                    break

        elif choice == '5':
            chunk_id = input("Enter Chunk ID: ").strip()
            if not chunk_id: continue
            
            # Get the embedding for this ID
            target = collection.get(ids=[chunk_id], include=["embeddings", "documents", "metadatas"])
            if not target["ids"]:
                print(f"❌ Chunk ID '{chunk_id}' not found.")
                continue
            
            embedding = target["embeddings"][0]
            print(f"\n🔍 Finding chunks similar to ID: {chunk_id}")
            print(f"Content: {target['documents'][0][:100]}...")
            
            # Query similar chunks
            results = collection.query(
                query_embeddings=[embedding],
                n_results=6, # Include itself
                include=["documents", "metadatas", "distances"]
            )
            
            print("\n📌 Most Similar Chunks:")
            for i in range(len(results["ids"][0])):
                res_id = results["ids"][0][i]
                if res_id == chunk_id:
                    continue # Skip itself
                
                dist = results["distances"][0][i]
                sim = 1 - dist # Cosine similarity
                doc = results["documents"][0][i]
                meta = results["metadatas"][0][i]
                
                print(f"\n[{i}] Similarity: {sim:.4f} (Distance: {dist:.4f})")
                print(f"    ID: {res_id}")
                print(f"    File: {meta.get('filename')} (Category: {meta.get('category')})")
                print(f"    Content: {doc[:200]}...")

        elif choice == 'q':
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()

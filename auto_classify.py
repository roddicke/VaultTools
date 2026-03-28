#!/usr/bin/env python3
"""
auto_classify.py - Automatically categorize files in the root of the Public vault
using a local LLM (Ollama) and move them to the appropriate subfolders.
"""

import os
import json
import shutil
import urllib.request
from pathlib import Path

# ─── Configuration ───────────────────────────────────────────────────────────

VAULT_PATH = Path("/Users/chaomingou/Public")
OLLAMA_MODEL = "qwen3:8b" # Or "mistral", "gemma", etc. 
CATEGORIES = [
    "00_Career",
    "10_Finance",
    "20_Medical",
    "30_Travel",
    "40_Resources",
    "50_Media",
    "60_Archive"
]

# Files to ignore in the root
IGNORE_FILES = {".DS_Store", ".localized", "Welcome.md"}
IGNORE_EXTENSIONS = {".canvas", ".base"} # Obsidian specific

# ─── LLM Classification ──────────────────────────────────────────────────────

def classify_file(filename: str, content_snippet: str = "") -> str:
    """Ask Ollama to classify a file into one of the categories."""
    
    # --- Keyword Fallback ---
    fn_lower = filename.lower()
    keyword_map = {
        "00_Career": ["career", "resume", "cv", "job", "work", "offer", "salary", "interview"],
        "10_Finance": ["bank", "transaction", "invoice", "receipt", "payment", "finance", "tax", "order", "purchase", "remittance", "expense", "費用", "送金", "振込", "領収書"],
        "20_Medical": ["medical", "doctor", "hospital", "health", "insurance", "clinical", "medication", "健康", "診察"],
        "30_Travel": ["travel", "flight", "hotel", "itinerary", "visa", "booking", "tours", "trip", "旅行", "航空券"],
        "40_Resources": ["guide", "manual", "book", "ref", "documentation", "resource", "data"],
        "tax2024": ["tax2024", "確定申告", "源泉徴収"],
        "永住": ["eijuu", "permanent residency", "permanent resident", "永住"]
    }
    
    for cat, keywords in keyword_map.items():
        if any(kw in fn_lower for kw in keywords):
            print(f"    ✨ Keyword match: {cat}")
            return cat

    prompt = f"""
Strictly classify the following file into one of these categories:
{", ".join(CATEGORIES)}

File Name: {filename}
Content Snippet: {content_snippet}

Respond ONLY with the category name. If unsure, respond with '40_Resources'.
Category:"""

    try:
        data = json.dumps({
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0}
        }).encode()
        
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=data,
            headers={"Content-Type": "application/json"}
        )
        
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode())
            category = result.get("response", "").strip()
            
            # Clean up the response in case LLM adds extra text
            for cat in CATEGORIES:
                if cat.lower() in category.lower():
                    return cat
            return "40_Resources"
    except Exception as e:
        print(f"  ⚠ Classification error for {filename}: {e}")
        return "40_Resources"

# ─── Extraction ──────────────────────────────────────────────────────────────

def get_content_snippet(filepath: Path, max_chars: int = 500) -> str:
    """Get a snippet of text from the file for better classification."""
    ext = filepath.suffix.lower()
    if ext in {".md", ".txt"}:
        try:
            return filepath.read_text(encoding="utf-8", errors="ignore")[:max_chars]
        except:
            return ""
    return ""

# ─── Main Logic ───────────────────────────────────────────────────────────────

def main():
    print("📂 Auto-Classification Started")
    print(f"   Scanning: {VAULT_PATH}")
    
    files_to_process = []
    for item in VAULT_PATH.iterdir():
        if item.is_file():
            if item.name in IGNORE_FILES:
                continue
            if item.suffix.lower() in IGNORE_EXTENSIONS:
                continue
            if item.name.startswith("."):
                continue
            files_to_process.append(item)
    
    if not files_to_process:
        print("  ✅ No new files found in root.")
        return

    print(f"  🔍 Found {len(files_to_process)} files to classify.\n")

    for filepath in files_to_process:
        print(f"📄 Processing: {filepath.name}")
        
        snippet = get_content_snippet(filepath)
        category = classify_file(filepath.name, snippet)
        
        target_dir = VAULT_PATH / category
        target_dir.mkdir(parents=True, exist_ok=True)
        
        target_path = target_dir / filepath.name
        
        # Avoid overwriting existing files
        if target_path.exists():
            print(f"  ⚠ Target already exists: {category}/{filepath.name} (skipping)")
            continue
            
        try:
            shutil.move(str(filepath), str(target_path))
            print(f"  ✅ Moved to: {category}/")
        except Exception as e:
            print(f"  ❌ Failed to move {filepath.name}: {e}")

    print("\n✨ Auto-classification complete!")

if __name__ == "__main__":
    main()

# VaultTools

Automated file organization and vector search for your Obsidian/Public vault.

## Features
- **Auto-Classify**: Automatically moves files from the vault root to subfolders using keywords and LLM (Ollama).
- **Auto-Sync**: Extracts text from documents and indexes them into ChromaDB.
- **Natural Language Query**: Ask questions about your vault in Japanese or Chinese.

## Setup Instructions

### 1. Install Prerequisites
- **Python 3**: Ensure you have Python 3.9+ installed.
- **Ollama**: Download and install from [ollama.com](https://ollama.com).
- **Pull Models**:
  ```bash
  ollama pull qwen3:8b
  ollama pull nomic-embed-text
  ```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Usage
- **Classify & Index**: `python3 auto_sync.py` (or run `auto_classify.py` first).
- **Query**: `python3 query.py "你的問題"`
- **Browse DB**: `python3 browse_db.py`

### 4. macOS Automation (Optional)
The system is pre-configured to run twice daily at **08:00** and **20:00**.
To enable it, copy `com.vaulttools.autosync.plist` to `~/Library/LaunchAgents/` and load it:
```bash
launchctl load ~/Library/LaunchAgents/com.vaulttools.autosync.plist
```
*(Note: Ensure `python3` is in your PATH and check Full Disk Access if it fails to read protected folders.)*

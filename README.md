# Vault AI — Private RAG Assistant

A privacy-first, offline AI assistant that answers questions grounded in your personal documents. Everything runs locally on your machine — no API keys, no cloud, no data leaves your computer.

Built with a three-service microservices architecture using FastAPI, ChromaDB, and Ollama.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Browser (GUI)                       │
│              Upload files · Ask questions               │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP
                       ▼
┌─────────────────────────────────────────────────────────┐
│              Orchestrator (FastAPI :8000)               │
│                                                         │
│  • Central API — single point of contact for the GUI    │
│  • Watches /vault folder for new files (watchdog)       │
│  • Routes queries to Ingestion → Inference              │
│  • Streams responses back to the browser                │
└──────────┬──────────────────────────────┬───────────────┘
           │                              │
           ▼                              ▼
┌────────────────────────┐  ┌────────────────────────────┐
│ Ingestion (8002)       │  │  Inference (8001)          │
│                        │  │                            │
│ • Extracts text from   │  │ • Takes user query +       │
│   PDFs and TXT files   │  │   retrieved snippets       │
│ • Chunks documents     │  │ • Feeds to local LLM       │
│   into paragraphs      │  │   via Ollama               │
│ • Generates vector     │  │ • Streams response         │
│   embeddings           │  │   token-by-token           │
│ • Stores in ChromaDB   │  │                            │
│ • Retrieves relevant   │  │                            │
│   chunks via similarity│  │                            │
│   search               │  │                            │
└────────────────────────┘  └────────────────────────────┘
```

---

## How it works

1. **Drop a file** into the vault (drag & drop via the GUI, or place directly in the `/vault` folder).
2. **Orchestrator** detects the new file via `watchdog` and triggers the Ingestion service.
3. **Ingestion** extracts text, chunks it into paragraphs with sentence overlap, generates vector embeddings using `all-MiniLM-L6-v2`, and stores them in ChromaDB.
4. **User asks a question** via the browser.
5. **Orchestrator** sends the query to Ingestion, which performs similarity search and returns the top 3 most relevant document chunks.
6. **Orchestrator** forwards the query + chunks to the Inference service.
7. **Inference** feeds the context and question to a local LLM via Ollama and streams the response token-by-token.
8. **Browser** displays the answer in real time as it generates.

---

## Tech stack


  Orchestrator : Python, FastAPI, watchdog, httpx 
  Ingestion : Python, FastAPI, PyMuPDF (fitz), NLTK, sentence-transformers, ChromaDB 
  Inference : Python, FastAPI, Ollama 
  Embedding model : all-MiniLM-L6-v2 (runs locally) 
  LLM : llama3.2:1b via Ollama (runs locally, swappable) 
  Vector database : ChromaDB (persistent, file-based) 
  Frontend : Single-file HTML/CSS/JS (no build tools) 

---

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed and running
- Pull a model: `ollama pull llama3.2:1b` (or `llama3.1` for better quality on powerful hardware)

### Installation

```bash
# Clone the repo
git clone https://github.com/mtyc888/personal_llm.git
cd vault-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install fastapi uvicorn httpx watchdog pydantic
pip install sentence-transformers chromadb nltk pymupdf
pip install ollama
```

### Running

Open three terminals:

```bash
# Terminal 1 — Orchestrator (port 8000)
cd orchestrator
uvicorn main:app --port 8000

# Terminal 2 — Inference (port 8001)
cd inference
uvicorn main:app --port 8001

# Terminal 3 — Ingestion (port 8002)
cd ingestion
uvicorn main:app --port 8002
```

Then open `frontend/index.html` in your browser (or serve it):

```bash
cd frontend
python -m http.server 3000
# Open http://localhost:3000
```

---

## Usage

1. Click **Vault** in the top right corner.
2. Drag and drop a PDF or TXT file. Wait for the status to show **ready**.
3. Close the vault panel.
4. Type a question about your documents and press Enter.
5. Watch the answer stream in real time.

---

## Project structure

```
vault-ai/
├── orchestrator/
│   └── main.py          # Central API, vault watcher, request routing
├── ingestion/
│   └── main.py          # Text extraction, chunking, embedding, vector storage
├── inference/
│   └── main.py          # LLM inference via Ollama with streaming
├── frontend/
│   └── chat.html       # Single-file browser GUI
├── vault/               # Drop documents here (auto-created)
└── README.md
```

---

## Design decisions

**Why microservices?** Each service has a distinct responsibility and can be developed, tested, and scaled independently. The ingestion service is CPU-bound (embedding), the inference service is GPU-bound (LLM), and the orchestrator is I/O-bound (routing). Separating them prevents a slow LLM generation from blocking file ingestion.

**Why local-only?** Privacy. Personal documents (journals, financial records, notes) should never leave the user's machine. Running Ollama locally means zero API keys, zero cloud dependency, and zero data leakage.

**Why ChromaDB?** Lightweight, file-based, no server setup required. Supports persistent storage and cosine similarity search out of the box. Ideal for a local application.

**Why streaming?** User experience. Without streaming, the user stares at a blank screen for 10-30 seconds while the LLM generates. Streaming shows tokens as they're produced, making the app feel responsive even on slower hardware.

---

## Limitations

- Supports PDF and TXT files only (no DOCX, EPUB, or images).
- No authentication or multi-user support (designed for personal use).
- Answer quality depends on the local LLM model size and hardware.
- No conversation memory across sessions (each query is independent).

---
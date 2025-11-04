# Film Search Engine (Vector Store + LLM)

Simple, modern movie search engine using local embeddings (sentence-transformers), FAISS vector store, and a clean API with a Streamlit UI.

## What You Get
- Natural language query parsing (genres, years, names) with spaCy/regex
- Semantic search with vector embeddings (all-MiniLM-L6-v2)
- Hybrid ranking (semantic + filters + ratings/popularity)
- FastAPI endpoint for integration
- Streamlit UI for demos

## Quick Start (with Poetry)

1) Install Poetry (if not already installed)
```
pip install poetry
```

2) Install project dependencies (creates and manages a virtualenv)
```
poetry install
```

3) Download the spaCy language model (one-time)
```
poetry run python -m spacy download en_core_web_sm
```

4) Build the vector index (one-time; creates models/faiss_index.*)
```
poetry run python scripts/build_index.py
```

5) Start the API server (auto-loads saved index if present)
```
poetry run uvicorn api:app --reload
```
- Health: http://localhost:8000/health
- Search: http://localhost:8000/search?q=sci-fi%20movies%20from%20the%2090s%20with%20Tom%20Hanks

6) Start the Streamlit UI (optional)
```
poetry run streamlit run streamlit_app.py
```
- By default, the UI will try the API; if unavailable, it will auto-load the local engine and saved index.

## Project Structure
```
.
├── data/
│   ├── movies.jsonl           # ~9,700 movies (used by scripts and API)
│   └── ...
├── models/
│   ├── faiss_index.index      # saved FAISS index (created by script)
│   └── faiss_index.pkl        # metadata (ids, mapping)
├── scripts/
│   └── build_index.py         # data script to build/save FAISS index
├── src/
│   ├── data_loader.py         # load + normalize data
│   ├── embeddings.py          # sentence-transformers integration
│   ├── vector_store.py        # FAISS index management
│   ├── query_parser.py        # spaCy NER + regex date/genre parsing
│   ├── ranking.py             # hybrid ranking logic
│   └── search_engine.py       # orchestrates parse → search → rank (loads saved index if present)
├── tests/
│   └── test_query_parser.py   # parser unit tests
├── api.py                     # FastAPI server exposing /search (loads saved index if present)
├── streamlit_app.py           # Streamlit UI calling the API
├── pyproject.toml             # Poetry configuration
└── README.md
```

## Useful Poetry Commands
- Install deps: `poetry install`
- Enter venv shell: `poetry shell`
- Run a Python script: `poetry run python scripts/build_index.py`
- Run API: `poetry run uvicorn api:app --reload`
- Run Streamlit UI: `poetry run streamlit run streamlit_app.py`
- Run tests: `poetry run python tests/test_query_parser.py`
- Install spaCy model: `poetry run python -m spacy download en_core_web_sm`

## Execution Notes
- First-time runs will download the embedding model (~90MB) and spaCy model.
- Run `scripts/build_index.py` once to enable fast API/UI startup via prebuilt index.
- If `models/faiss_index.*` exist, the API/UI will load them; otherwise they will build an index on demand.

## Troubleshooting
- If running outside `poetry shell`, always prefix commands with `poetry run ...`.
- If the API is slow on first request, it may be loading models; subsequent queries are fast.
- Windows PowerShell: use `;` to chain commands.

## Roadmap (Optional Enhancements)
- Tune ranking weights and add fuzzy name matching
- Add caching for repeated queries
- Add authentication/rate limiting if exposing publicly

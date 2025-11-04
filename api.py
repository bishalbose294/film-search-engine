"""
FastAPI server exposing the movie search API.
Endpoints:
- GET /health: basic health check
- GET /search?q=...&top_k=10: returns ranked results with scores and metadata

Startup loads a saved FAISS index if available (models/faiss_index.*),
otherwise builds embeddings and indexes on the fly.
"""

# Import standard libraries for filesystem paths and timing
import os  # not strictly required but kept for future env-based settings
import time  # measure startup and request latencies
from typing import List, Optional  # precise typing for clarity
from pathlib import Path  # path-safe filesystem handling

# Import FastAPI for building the web API and Pydantic for response models
from fastapi import FastAPI, Query  # FastAPI primitives
from pydantic import BaseModel  # response schema definitions

# Import our internal modules for data loading and search
from src.data_loader import DataLoader  # loads and normalizes movies
from src.search_engine import SearchEngine, SearchResult  # core search engine

# Import loguru for simple, structured console logging
from loguru import logger  # convenient console logger

# Instantiate the FastAPI application with metadata
app = FastAPI(title="Film Search Engine API", version="1.0.0")  # web app

# Globals that hold the search engine instance and measured startup time
ENGINE: Optional[SearchEngine] = None  # will point to the initialized engine
STARTUP_TIME_S: float = 0.0  # measures how long startup took


# Pydantic model that describes the shape of a single movie in responses
class MovieOut(BaseModel):
	id: str  # unique id
	title: str  # human-readable title
	year: int  # release year
	genres: List[str]  # list of genres
	director: Optional[str] = None  # director name if present
	actors: List[str]  # subset of actors for brevity
	rating: float  # average rating
	popularity: float  # popularity score
	poster_url: Optional[str] = None  # optional poster image URL
	overview: Optional[str] = None  # short synopsis snippet


# Pydantic model for a single ranked search item
class SearchResponseItem(BaseModel):
	movie: MovieOut  # movie metadata
	score: float  # final ranking score
	similarity: float  # raw vector similarity


# Pydantic model for the complete search response payload
class SearchResponse(BaseModel):
	query: str  # original query string
	top_k: int  # number of results requested
	elapsed_ms: float  # server-side search time in ms
	results: List[SearchResponseItem]  # ranked items


# FastAPI startup hook to initialize the search engine once
@app.on_event("startup")
async def startup_event():
	"""Initialize the search engine and log how it was initialized."""
	global ENGINE, STARTUP_TIME_S  # refer to module-level globals
	start = time.time()  # start timer for startup latency

	logger.info("[API] Startup: loading movies and initializing engine...")  # log intent

	# Load movies from JSONL using the shared DataLoader
	loader = DataLoader()  # create loader instance
	movies = loader.load_movies_from_jsonl('data/movies.jsonl')  # read dataset
	logger.info(f"[API] Loaded {len(movies)} movies for indexing/search")  # record dataset size

	# Detect if a prebuilt FAISS index exists to speed up startup
	index_base = Path('models') / 'faiss_index'  # base path (without extension)
	index_exists = index_base.with_suffix('.index').exists() and index_base.with_suffix('.pkl').exists()  # both files present?
	load_path = str(index_base) if index_exists else None  # pass to engine if present

	# Construct SearchEngine; it will load saved index or build anew based on load_path
	ENGINE = SearchEngine(movies, load_index_base_path=load_path)  # create engine

	# Compute and log startup duration and mode
	STARTUP_TIME_S = time.time() - start  # elapsed seconds
	mode = 'Loaded saved index' if load_path else 'Built new index'  # mode string
	logger.info(f"[API] Startup complete in {STARTUP_TIME_S:.2f}s. {mode}.")  # summary log


# Simple health endpoint for readiness checks
@app.get("/health")
async def health():
	"""Return minimal health info for liveness/readiness probes."""
	return {
		"status": "ok",  # constant indicator
		"engine_ready": ENGINE is not None,  # True if engine initialized
		"startup_seconds": round(STARTUP_TIME_S, 2)  # startup latency
	}


# Main search endpoint that accepts a free-text query
@app.get("/search", response_model=SearchResponse)
async def search(q: str = Query(..., description="Natural language movie query"), top_k: int = 10):
	"""Execute a semantic search and return ranked results."""
	if ENGINE is None:  # engine must be ready to serve
		logger.warning("[API] Search requested but engine not initialized")  # guard log
		return SearchResponse(query=q, top_k=top_k, elapsed_ms=0.0, results=[])  # return empty

	# Time the search for latency insight
	start = time.time()  # start timer
	logger.debug(f"[API] /search q='{q}' top_k={top_k}")  # debug log of input

	# Delegate to the engine
	results = ENGINE.search(q, top_k=top_k)  # run search
	elapsed_ms = (time.time() - start) * 1000  # compute ms
	logger.info(f"[API] /search served {len(results)} results in {elapsed_ms:.2f} ms")  # summary

	# Convert engine results to response schema
	items: List[SearchResponseItem] = []  # accumulator
	for r in results:  # iterate ranked results
		m = r.movie  # movie object
		items.append(  # append converted item
			SearchResponseItem(
				movie=MovieOut(
					id=m.id,
					title=m.title.title(),
					year=m.year,
					genres=m.genres,
					director=m.director,
					actors=m.actors[:5],
					rating=m.rating,
					popularity=m.popularity,
					poster_url=m.poster_url,
					overview=m.overview[:350] if m.overview else None,
				),
				score=round(r.score, 3),
				similarity=round(r.similarity, 3),
			)
		)

	# Return structured response with timing
	return SearchResponse(query=q, top_k=top_k, elapsed_ms=round(elapsed_ms, 2), results=items)

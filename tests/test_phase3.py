"""
Test script for Phase 3: Query Understanding with spaCy NER and regex.
Parses a variety of natural language queries and prints the extracted structure.
Optionally performs a quick semantic search using previously built vector store.
"""

import time
from pathlib import Path

from src.data_loader import DataLoader
from src.query_parser import QueryParser
from src.search_engine import SearchEngine


def ensure_models_dir():
    p = Path('models')
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)


def build_engine(movies):
    """Create SearchEngine that validates/loads or builds/saves index."""
    ensure_models_dir()
    index_base = 'models/faiss_index'
    engine = SearchEngine(movies, load_index_base_path=index_base)
    return engine


def demo_search(engine: SearchEngine, raw_query: str):
    print(f"\nVector search demo for: '{raw_query}'")
    results = engine.search(raw_query, top_k=5)
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r.similarity:.3f}] {r.movie.title.title()} ({r.movie.year}) - {', '.join(r.movie.genres[:3])}")


def test_phase3():
	print("="*60)
	print("Phase 3: Query Understanding Test")
	print("="*60)

	# Load dataset
	loader = DataLoader()
	movies = loader.load_movies_from_jsonl('data/movies.jsonl')
	actors = set(loader.get_all_actors(movies))
	directors = set(loader.get_all_directors(movies))

	# Initialize parser
	print("\n[Step 1] Initializing Query Parser...")
	parser = QueryParser(known_actors=actors, known_directors=directors)
	print("[OK] Parser ready")

	# Sample queries
	queries = [
		"sci-fi movies from the 90s with Tom Hanks",
		"comedy films in the 80s starring Eddie Murphy",
		"horror movies directed by Wes Craven",
		"funny films from the early 2000s",
		"war movies about love before 2000",
		"action movies after 2015 with keanu reeves"
	]

	# Parse and display
	print("\n[Step 2] Parsing sample queries...")
	for q in queries:
		pq = parser.parse(q)
		print(f"\nQuery: {q}")
		print(f"  Genres     : {pq.genres}")
		print(f"  Actors     : {pq.actors[:5]}")
		print(f"  Directors  : {pq.directors[:5]}")
		print(f"  Year Range : {pq.year_range}")
		print(f"  Keywords   : {pq.keywords}")

	# Optional: quick semantic search demo for first two queries
	print("\n[Step 3] Optional vector search demo (load-or-build index)...")
	start = time.time()
	engine = build_engine(movies)
	print(f"[OK] Engine ready in {time.time() - start:.2f}s")

	for q in queries[:2]:
		demo_search(engine, q)

	print("\n" + "="*60)
	print("Phase 3 complete!")
	print("="*60)


if __name__ == '__main__':
	test_phase3()

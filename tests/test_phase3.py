"""
Test script for Phase 3: Query Understanding with spaCy NER and regex.
Parses a variety of natural language queries and prints the extracted structure.
Optionally performs a quick semantic search using previously built vector store.
"""

import time
from pathlib import Path

from src.data_loader import DataLoader
from src.query_parser import QueryParser
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore


def ensure_models_dir():
	p = Path('models')
	if not p.exists():
		p.mkdir(parents=True, exist_ok=True)


def build_vector_store(movies):
	"""Build vector store (FAISS) from movies; used for demo search."""
	emb = EmbeddingGenerator('all-MiniLM-L6-v2')
	movie_embeddings = emb.generate_movie_embeddings(movies, batch_size=32, show_progress=True)
	store = VectorStore(emb.get_embedding_dimension(), use_cosine_similarity=True)
	store.add_movies(movies, movie_embeddings)
	return emb, store


def demo_search(embedding_gen, vector_store, raw_query):
	print(f"\nVector search demo for: '{raw_query}'")
	q_emb = embedding_gen.generate_query_embedding(raw_query)
	results = vector_store.search(q_emb, top_k=5)
	for i, (mid, score) in enumerate(results, 1):
		m = vector_store.get_movie_by_id(mid)
		if m:
			print(f"  {i}. [{score:.3f}] {m.title.title()} ({m.year}) - {', '.join(m.genres[:3])}")


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
	print("\n[Step 3] Optional vector search demo (building index)...")
	start = time.time()
	emb, store = build_vector_store(movies)
	print(f"[OK] Built vector store in {time.time() - start:.2f}s")

	for q in queries[:2]:
		demo_search(emb, store, q)

	print("\n" + "="*60)
	print("Phase 3 complete!")
	print("="*60)


if __name__ == '__main__':
	test_phase3()

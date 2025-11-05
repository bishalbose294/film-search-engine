"""
Phase 4 Test: Hybrid Ranking and Vector Search
Runs end-to-end search using query parser, embeddings, vector store, and ranker.
"""

import time
from src.data_loader import DataLoader
from src.search_engine import SearchEngine


def test_phase4():
	print("="*60)
	print("Phase 4: Hybrid Ranking & Vector Search Test")
	print("="*60)

	loader = DataLoader()
	movies = loader.load_movies_from_jsonl('data/movies.jsonl')

	print("\n[Step 1] Building/Loading Search Engine (validates index)...")
	start = time.time()
	engine = SearchEngine(movies, load_index_base_path='models/faiss_index')
	build_time = time.time() - start
	print(f"[OK] Search engine built in {build_time:.2f}s")

	queries = [
		"sci-fi movies from the 90s with Tom Hanks",
		"comedy films in the 80s starring Eddie Murphy",
		"horror movies directed by Wes Craven",
		"funny films from the early 2000s",
		"war movies about love before 2000",
		"action movies after 2015 with keanu reeves"
	]

	print("\n[Step 2] Executing searches with hybrid ranking...")
	for q in queries:
		start = time.time()
		results = engine.search(q, top_k=5)
		elapsed = (time.time() - start) * 1000
		print(f"\nQuery: {q}")
		print(f"  Time: {elapsed:.2f} ms | Results: {len(results)}")
		for i, r in enumerate(results, 1):
			print(f"  {i}. [score {r.score:.3f} | sim {r.similarity:.3f}] {r.movie.title.title()} ({r.movie.year}) - {', '.join(r.movie.genres[:3])}")

	print("\n" + "="*60)
	print("Phase 4 complete!")
	print("="*60)


if __name__ == '__main__':
	test_phase4()

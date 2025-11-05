"""
Test script for Phase 2: Vector Store Setup & Embedding Generation
Verifies that embedding generation and vector store work correctly.
"""

import time
from src.data_loader import DataLoader
from src.search_engine import SearchEngine

def test_phase2():
    """Test embedding generation and vector store setup."""
    print("="*60)
    print("Phase 2: Vector Store & Embedding Generation Test")
    print("="*60)
    
    # Step 1: Load movies
    print("\n[Step 1] Loading movies...")
    loader = DataLoader()
    movies = loader.load_movies_from_jsonl('data/movies.jsonl')
    print(f"[OK] Loaded {len(movies)} movies")
    
    # Step 2: Initialize/load SearchEngine (validates existing index or builds/saves once)
    print("\n[Step 2] Initializing SearchEngine (load-or-build index)...")
    start_time = time.time()
    engine = SearchEngine(movies, load_index_base_path='models/faiss_index')
    engine_time = time.time() - start_time
    print(f"[OK] Engine ready in {engine_time:.2f} seconds")
    
    # Step 3: Test search functionality
    print("\n[Step 3] Testing vector search...")
    test_queries = [
        "sci-fi movies with space travel",
        "funny comedy films",
        "horror movies about ghosts"
    ]
    
    for query in test_queries:
        print(f"\n  Query: '{query}'")
        start_time = time.time()
        
        # Search
        results = engine.search(query, top_k=5)
        
        search_time = time.time() - start_time
        
        print(f"  Search time: {search_time*1000:.2f} ms")
        print(f"  Top 5 results:")
        for i, r in enumerate(results, 1):
            print(f"    {i}. [{r.similarity:.3f}] {r.movie.title.title()} ({r.movie.year}) - {', '.join(r.movie.genres[:3])}")
    
    # Summary
    print("\n" + "="*60)
    print("Phase 2 Test Summary")
    print("="*60)
    print(f"Total time: {engine_time:.2f} seconds")
    print(f"  - Engine init/load/build: {engine_time:.2f}s")
    print(f"\n[OK] Phase 2 complete! Engine is ready for search.")
    print("="*60)

if __name__ == '__main__':
    test_phase2()

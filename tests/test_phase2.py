"""
Test script for Phase 2: Vector Store Setup & Embedding Generation
Verifies that embedding generation and vector store work correctly.
"""

import time
from src.data_loader import DataLoader
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore

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
    
    # Step 2: Initialize embedding generator
    print("\n[Step 2] Initializing embedding model...")
    start_time = time.time()
    embedding_generator = EmbeddingGenerator(model_name='all-MiniLM-L6-v2')
    init_time = time.time() - start_time
    print(f"[OK] Model initialized in {init_time:.2f} seconds")
    print(f"     Embedding dimension: {embedding_generator.get_embedding_dimension()}")
    
    # Step 3: Generate embeddings for movies
    print("\n[Step 3] Generating embeddings for all movies...")
    start_time = time.time()
    movie_embeddings = embedding_generator.generate_movie_embeddings(
        movies, 
        batch_size=32,
        show_progress=True
    )
    embedding_time = time.time() - start_time
    print(f"[OK] Generated embeddings in {embedding_time:.2f} seconds")
    print(f"     Embedding shape: {movie_embeddings.shape}")
    
    # Step 4: Create vector store
    print("\n[Step 4] Creating FAISS vector store...")
    start_time = time.time()
    vector_store = VectorStore(
        embedding_dimension=embedding_generator.get_embedding_dimension(),
        use_cosine_similarity=True
    )
    vector_store.add_movies(movies, movie_embeddings)
    index_time = time.time() - start_time
    print(f"[OK] Vector store created in {index_time:.2f} seconds")
    print(f"     Total movies in index: {vector_store.size()}")
    
    # Step 5: Test search functionality
    print("\n[Step 5] Testing vector search...")
    test_queries = [
        "sci-fi movies with space travel",
        "funny comedy films",
        "horror movies about ghosts"
    ]
    
    for query in test_queries:
        print(f"\n  Query: '{query}'")
        start_time = time.time()
        
        # Generate query embedding
        query_embedding = embedding_generator.generate_query_embedding(query)
        
        # Search
        results = vector_store.search(query_embedding, top_k=5)
        
        search_time = time.time() - start_time
        
        print(f"  Search time: {search_time*1000:.2f} ms")
        print(f"  Top 5 results:")
        for i, (movie_id, score) in enumerate(results, 1):
            movie = vector_store.get_movie_by_id(movie_id)
            if movie:
                print(f"    {i}. [{score:.3f}] {movie.title.title()} ({movie.year}) - {', '.join(movie.genres[:3])}")
    
    # Step 6: Test saving and loading (optional)
    print("\n[Step 6] Testing index save/load...")
    try:
        vector_store.save_index('models/faiss_index')
        print("[OK] Index saved successfully")
        
        # Load it back
        loaded_store = VectorStore.load_index('models/faiss_index', movies=movies)
        print(f"[OK] Index loaded successfully - {loaded_store.size()} movies")
        
        # Quick test with loaded index
        test_query = "action movies"
        query_emb = embedding_generator.generate_query_embedding(test_query)
        results = loaded_store.search(query_emb, top_k=3)
        print(f"[OK] Search works with loaded index - found {len(results)} results")
    except Exception as e:
        print(f"[WARNING] Save/load test failed: {e}")
        print("         (This is optional, continuing...)")
    
    # Summary
    print("\n" + "="*60)
    print("Phase 2 Test Summary")
    print("="*60)
    print(f"Total time: {init_time + embedding_time + index_time:.2f} seconds")
    print(f"  - Model initialization: {init_time:.2f}s")
    print(f"  - Embedding generation: {embedding_time:.2f}s")
    print(f"  - Index creation: {index_time:.2f}s")
    print(f"\n[OK] Phase 2 complete! Vector store is ready for search.")
    print("="*60)

if __name__ == '__main__':
    test_phase2()

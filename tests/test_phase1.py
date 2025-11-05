"""
Test script for Phase 1: Data Loading
Verifies that data loading and preprocessing works correctly.
"""

from src.data_loader import DataLoader

def test_data_loading():
    """Test loading movies from JSONL file."""
    loader = DataLoader()
    
    # Load movies
    movies = loader.load_movies_from_jsonl('data/movies.jsonl')
    
    print(f"\n{'='*60}")
    print(f"Data Loading Test Results")
    print(f"{'='*60}")
    print(f"Total movies loaded: {len(movies)}")
    
    if movies:
        # Show first movie as example
        print(f"\nFirst movie example:")
        print(f"  ID: {movies[0].id}")
        print(f"  Title: {movies[0].title}")
        print(f"  Genres: {movies[0].genres}")
        print(f"  Director: {movies[0].director}")
        print(f"  Actors (first 3): {movies[0].actors[:3]}")
        print(f"  Year: {movies[0].year}")
        print(f"  Rating: {movies[0].rating}")
        print(f"\nSearchable text (first 200 chars):")
        print(f"  {movies[0].searchable_text[:200]}...")
        
        # Statistics
        all_genres = loader.get_all_genres(movies)
        all_actors = loader.get_all_actors(movies)
        all_directors = loader.get_all_directors(movies)
        
        print(f"\nDataset Statistics:")
        print(f"  Unique genres: {len(all_genres)}")
        print(f"  Unique actors: {len(all_actors)}")
        print(f"  Unique directors: {len(all_directors)}")
        print(f"  Year range: {min(m.year for m in movies if m.year)} - {max(m.year for m in movies if m.year)}")
        
    print(f"\n{'='*60}")
    print("Phase 1 Test Complete!")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    test_data_loading()


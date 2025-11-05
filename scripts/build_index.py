"""
Build and persist the vector store index.

This script:
1) Loads movies from data/movies.jsonl
2) Generates embeddings (sentence-transformers)
3) Builds a FAISS index
4) Saves the index and metadata to models/

Usage:
    poetry run python -m scripts.build_index

After running this once, the API and Streamlit can load the saved index
for faster startup.
"""

import time  # measure step timings
from pathlib import Path  # filesystem-safe paths

from loguru import logger  # console logging

from src.data_loader import DataLoader  # data ingestion
from src.embeddings import EmbeddingGenerator  # embedding model wrapper
from src.vector_store import VectorStore  # FAISS index helper


def main():
	# Headline banner for visibility in console
	logger.info("=" * 60)
	logger.info("Build Vector Store Index")
	logger.info("=" * 60)

	# Resolve project root and key paths
	root = Path(__file__).resolve().parents[1]  # project root
	data_path = root / 'data' / 'movies.jsonl'  # input dataset
	models_dir = root / 'models'  # output directory for index files
	models_dir.mkdir(parents=True, exist_ok=True)  # ensure exists
	index_base = models_dir / 'faiss_index'  # base filename (no extension)

	# 1) Load data
	logger.info("[1/4] Loading movies...")
	loader = DataLoader()  # loader instance
	movies = loader.load_movies_from_jsonl(str(data_path))  # read dataset
	logger.info(f"[OK] Loaded {len(movies)} movies")  # confirm count

	# 2) Generate embeddings
	logger.info("\n[2/4] Generating embeddings...")
	t0 = time.time()  # start timer
	emb = EmbeddingGenerator('all-MiniLM-L6-v2')  # load model
	movie_embeddings = emb.generate_movie_embeddings(movies, batch_size=32, show_progress=True)  # matrix
	logger.info(f"[OK] Embeddings generated in {time.time() - t0:.2f}s; shape={movie_embeddings.shape}")  # report

	# 3) Build FAISS index
	logger.info("\n[3/4] Building FAISS index...")
	store = VectorStore(emb.get_embedding_dimension(), use_cosine_similarity=True)  # create index
	store.add_movies(movies, movie_embeddings)  # add vectors
	logger.info(f"[OK] Index built with {store.size()} vectors")  # report size

	# 4) Save index
	logger.info("\n[4/4] Saving index and metadata...")
	store.save_index(str(index_base))  # write .index and .pkl
	logger.info("[OK] Saved.")  # done

	# Footer
	logger.info("\nAll done! The API/Streamlit will load this saved index if present.")
	logger.info("=" * 60)


if __name__ == '__main__':
	main()  # invoke builder

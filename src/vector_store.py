"""
Vector store module using FAISS.
Handles creating, managing, and searching the FAISS index for movie embeddings.
"""

# Import NumPy for typed arrays passed to FAISS
import numpy as np  # numeric arrays
# Import FAISS (Facebook AI Similarity Search) for fast nearest-neighbor search
import faiss  # vector index
# Pathlib for robust path handling when saving/loading
from pathlib import Path  # filesystem paths
# Typing hints for clarity of public API
from typing import List, Tuple, Optional  # type hints
# Pickle for persisting small Python metadata (IDs/mappings)
import pickle  # simple serialization

# Import our Movie model for type hints and mapping
from .models import Movie  # movie data class

# Console logging
from loguru import logger  # console logger


class VectorStore:
	"""
	Manages FAISS vector store for fast similarity search.
	"""
	
	def __init__(self, embedding_dimension: int, use_cosine_similarity: bool = True):
		"""
		Initialize the FAISS index for a specific embedding dimension and metric.
		- embedding_dimension: length of each embedding vector (e.g., 384)
		- use_cosine_similarity: choose cosine (IP on normalized vectors) or L2 distance
		"""
		# Save configuration for later checks and persistence
		self.embedding_dimension = embedding_dimension  # vector length
		self.use_cosine_similarity = use_cosine_similarity  # metric flag
		self.index = None  # will hold the FAISS index object
		self.movie_ids = []  # list mapping index row -> movie_id
		self.movies_map = {}  # dict mapping movie_id -> Movie object for retrieval
		
		# Create a flat (non-compressed) FAISS index with the chosen metric
		if use_cosine_similarity:
			# Inner product on normalized vectors equals cosine similarity
			self.index = faiss.IndexFlatIP(embedding_dimension)
		else:
			# L2 distance index (exact search)
			self.index = faiss.IndexFlatL2(embedding_dimension)
		
		logger.info(
			f"[VectorStore] Initialized FAISS index | dim={embedding_dimension} | metric={'cosine' if use_cosine_similarity else 'l2'}"
		)
	
	def add_movies(self, movies: List[Movie], embeddings: np.ndarray):
		"""
		Add movies and their embeddings to the vector store.
		- movies: list of Movie objects
		- embeddings: NumPy array of shape (num_movies, embedding_dimension)
		"""
		# Validate count consistency between metadata and vectors
		if len(movies) != embeddings.shape[0]:
			raise ValueError(
				f"Number of movies ({len(movies)}) doesn't match number of embeddings ({embeddings.shape[0]})"
			)
		# Validate embedding dimensionality
		if embeddings.shape[1] != self.embedding_dimension:
			raise ValueError(
				f"Embedding dimension ({embeddings.shape[1]}) doesn't match expected ({self.embedding_dimension})"
			)
		
		# Ensure FAISS receives float32 arrays (its preferred dtype)
		embeddings = embeddings.astype('float32')
		
		# Add vectors to the FAISS index
		logger.info(f"[VectorStore] Adding {len(movies)} vectors to index...")
		self.index.add(embeddings)
		
		# Record IDs and build a quick lookup for movie metadata
		for movie in movies:
			self.movie_ids.append(movie.id)
			self.movies_map[movie.id] = movie
		
		logger.info(f"[VectorStore] Added {len(movies)} movies | total in index: {self.index.ntotal}")
	
	def search(
		self, 
		query_embedding: np.ndarray, 
		top_k: int = 10
	) -> List[Tuple[str, float]]:
		"""
		Search for the top-k nearest movies to the given query embedding.
		Returns a list of (movie_id, similarity) pairs.
		"""
		# Ensure the index is populated before searching
		if self.index is None or self.index.ntotal == 0:
			raise ValueError("Vector store is empty. Add movies first.")
		
		# Convert to float32 as expected by FAISS and ensure 2D shape
		query_embedding = query_embedding.astype('float32')
		if query_embedding.ndim == 1:
			query_embedding = query_embedding.reshape(1, -1)
		
		# Validate vector length
		if query_embedding.shape[1] != self.embedding_dimension:
			raise ValueError(
				f"Query embedding dimension ({query_embedding.shape[1]}) doesn't match expected ({self.embedding_dimension})"
			)
		
		# Normalize in-place for cosine similarity (IP on unit vectors)
		if self.use_cosine_similarity:
			faiss.normalize_L2(query_embedding)
		
		# Execute the nearest-neighbor search
		distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
		
		# Translate FAISS row indices back to movie IDs and convert distances to scores
		results = []
		for distance, idx in zip(distances[0], indices[0]):
			if idx < 0:  # -1 indicates an invalid slot (no result)
				continue
			movie_id = self.movie_ids[idx]  # recover movie id
			if self.use_cosine_similarity:
				score = float(distance)  # already similarity
			else:
				# Convert L2 distance to a bounded similarity [0,1]
				score = float(1.0 / (1.0 + distance))
			results.append((movie_id, score))  # collect pair
		
		return results  # list of results
	
	def get_movie_by_id(self, movie_id: str) -> Optional[Movie]:
		"""Return a Movie object for a given ID, or None if not found."""
		return self.movies_map.get(movie_id)
	
	def get_movies_by_ids(self, movie_ids: List[str]) -> List[Movie]:
		"""Return a list of Movie objects given their IDs (missing become None)."""
		return [self.movies_map.get(movie_id) for movie_id in movie_ids]
	
	def size(self) -> int:
		"""Return the number of vectors currently stored in the index."""
		return self.index.ntotal if self.index else 0
	
	def save_index(self, filepath: str):
		"""
		Persist the FAISS index and essential metadata next to it.
		- filepath: base path without extension; we write .index and .pkl files
		"""
		filepath = Path(filepath)  # coerce to Path
		# Write the FAISS binary index
		index_path = filepath.with_suffix('.index')
		faiss.write_index(self.index, str(index_path))
		# Prepare and save Python metadata for IDs and settings
		metadata_path = filepath.with_suffix('.pkl')
		metadata = {
			'movie_ids': self.movie_ids,  # row -> id mapping
			'movies_map': self.movies_map,  # id -> Movie mapping (optional sized object)
			'embedding_dimension': self.embedding_dimension,  # config
			'use_cosine_similarity': self.use_cosine_similarity  # config
		}
		with open(metadata_path, 'wb') as f:
			pickle.dump(metadata, f)  # serialize metadata
		logger.info(f"[VectorStore] Saved index to {index_path} and metadata to {metadata_path}")
	
	@classmethod
	def load_index(
		cls, 
		filepath: str, 
		movies: Optional[List[Movie]] = None
	) -> 'VectorStore':
		"""
		Load a previously saved FAISS index and its metadata.
		- filepath: base path without extension (.index/.pkl inferred)
		- movies: optional fresh Movie list to rebuild the movies_map accurately
		"""
		filepath = Path(filepath)  # coerce to Path
		index_path = filepath.with_suffix('.index')  # FAISS file
		metadata_path = filepath.with_suffix('.pkl')  # Python metadata
		
		# Validate presence of both files
		if not index_path.exists():
			raise FileNotFoundError(f"Index file not found: {index_path}")
		if not metadata_path.exists():
			raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
		
		# Load FAISS index from disk
		index = faiss.read_index(str(index_path))
		# Load corresponding metadata
		with open(metadata_path, 'rb') as f:
			metadata = pickle.load(f)
		
		# Construct a new VectorStore with the saved configuration
		vector_store = cls(
			embedding_dimension=metadata['embedding_dimension'],
			use_cosine_similarity=metadata['use_cosine_similarity']
		)
		vector_store.index = index  # attach loaded index
		vector_store.movie_ids = metadata['movie_ids']  # restore id list
		
		# Rebuild movie mapping either from provided list or from saved dict
		if movies:
			vector_store.movies_map = {movie.id: movie for movie in movies}
		else:
			vector_store.movies_map = metadata.get('movies_map', {})
		
		logger.info(f"[VectorStore] Loaded index from {index_path} | total={vector_store.index.ntotal}")
		return vector_store  # ready-to-use store


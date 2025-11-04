"""
Embedding generation module.
Handles creating vector embeddings for movies and queries using sentence-transformers.
"""

# Import NumPy for numerical arrays that store embeddings
import numpy as np  # efficient numeric arrays
# Import typing helpers for clear API contracts
from typing import List  # list types
# Import the SentenceTransformer model to convert text into embeddings
from sentence_transformers import SentenceTransformer  # pre-trained embedding model

# Import our Movie data class so we can type and validate inputs
from .models import Movie  # structured movie object

# Import loguru for consistent console logging (friendlier than print)
from loguru import logger  # console logger


class EmbeddingGenerator:
	"""
	Generates embeddings for movies and queries using sentence-transformers.
	"""
	
	def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
		"""
		Initialize the embedding generator with a chosen sentence transformer model.
		- model_name selects which pre-trained model to load. We use a fast, accurate default.
		"""
		logger.info(f"[Embeddings] Loading embedding model: {model_name}")  # log model selection
		# Create the sentence transformer; downloads on first use then caches locally
		self.model = SentenceTransformer(model_name)  # load model weights
		# Keep the name for reference/diagnostics
		self.model_name = model_name  # save model id
		# Ask the model for the dimensionality of produced vectors (e.g., 384)
		self.embedding_dimension = self.model.get_sentence_embedding_dimension()  # vector size
		logger.info(f"[Embeddings] Model ready. Embedding dimension: {self.embedding_dimension}")  # confirm
	
	def generate_movie_embeddings(
		self, 
		movies: List[Movie], 
		batch_size: int = 32,
		show_progress: bool = True
	) -> np.ndarray:
		"""
		Generate embeddings for all movies by feeding their 'searchable_text' into the model.
		- movies: list of Movie objects prepared by the data loader
		- batch_size: number processed at a time for speed/memory balance
		- show_progress: whether to show a progress bar while computing
		Returns a NumPy array of shape (num_movies, embedding_dimension).
		"""
		# Guard against accidental empty input which would otherwise fail later
		if not movies:
			raise ValueError("No movies provided for embedding generation")  # explicit error
		
		# Collect the weighted searchable text for each movie (title/genres/actors/overview)
		texts = []  # will hold one string per movie
		for movie in movies:  # iterate dataset
			# Ensure preprocessing created the searchable text we rely on
			if not movie.searchable_text:
				raise ValueError(f"Movie {movie.id} missing searchable_text. Run preprocessing first.")
			# Append to list in the same order as movies so rows align with inputs
			texts.append(movie.searchable_text)
		
		logger.info(f"[Embeddings] Generating embeddings for {len(movies)} movies (batch {batch_size})")  # progress
		
		# Ask the model to encode all texts into normalized vectors (cosine-ready)
		embeddings = self.model.encode(
			texts,  # input documents
			batch_size=batch_size,  # batch size for efficiency
			show_progress_bar=show_progress,  # display progress bar
			convert_to_numpy=True,  # return as NumPy array
			normalize_embeddings=True  # L2-normalize so cosine == dot product
		)
		
		logger.info(f"[Embeddings] Generated matrix with shape {embeddings.shape}")  # summary
		return embeddings  # return embedding matrix
	
	def generate_query_embedding(self, query: str) -> np.ndarray:
		"""
		Generate an embedding vector for a single user query string.
		Returns a NumPy array of length 'embedding_dimension'.
		"""
		# Validate the query to avoid confusing errors downstream
		if not query or not query.strip():  # empty or whitespace only
			raise ValueError("Query cannot be empty")  # clear feedback
		
		# Feed the cleaned query into the model to obtain a normalized vector
		embedding = self.model.encode(
			query.strip(),  # trim surrounding spaces
			convert_to_numpy=True,  # NumPy vector
			normalize_embeddings=True  # normalized for cosine similarity
		)
		
		return embedding  # single vector
	
	def generate_query_embeddings(self, queries: List[str]) -> np.ndarray:
		"""
		Generate embeddings for multiple queries at once.
		Returns an array of shape (num_queries, embedding_dimension).
		"""
		# Guard: require at least one query string
		if not queries:
			raise ValueError("No queries provided")  # inform caller
		
		# Encode all queries together for better throughput
		embeddings = self.model.encode(
			queries,  # list of strings
			convert_to_numpy=True,  # NumPy matrix
			normalize_embeddings=True  # normalized rows
		)
		
		return embeddings  # matrix of vectors
	
	def get_embedding_dimension(self) -> int:
		"""
		Return the dimensionality of the embedding vectors produced by the model.
		"""
		return self.embedding_dimension  # cached value


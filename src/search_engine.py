"""
Search engine module.
Executes semantic vector search, applies filters, and produces ranked results.
"""

from dataclasses import dataclass  # lightweight containers for results
from typing import List, Optional, Tuple  # type annotations for clarity
from pathlib import Path  # used for optional index presence checks

import numpy as np  # kept for future array ops if needed

# Import project modules for data structures and components
from .models import Movie, ParsedQuery  # core data classes
from .embeddings import EmbeddingGenerator  # sentence-transformers wrapper
from .vector_store import VectorStore  # FAISS index manager
from .ranking import Ranker  # hybrid ranking logic
from .query_parser import QueryParser  # query understanding

# Import loguru for console logging
from loguru import logger  # simple structured logger


@dataclass
class SearchResult:
	movie: Movie  # matched movie
	score: float  # final ranking score
	similarity: float  # semantic similarity from vector search


class SearchEngine:
	"""
	High-level search API combining query parsing, vector search, filtering, and ranking.
	Optionally loads a prebuilt FAISS index from disk for fast startup.
	"""
	def __init__(
		self,
		movies: List[Movie],  # dataset of movies
		embedding_model_name: str = 'all-MiniLM-L6-v2',  # embedding model
		use_cosine_similarity: bool = True,  # FAISS similarity metric
		load_index_base_path: Optional[str] = None,  # e.g., 'models/faiss_index'
	):
		# Store movies for use across the pipeline
		self.movies = movies  # keep dataset reference
		# Initialize embedding generator (will load model once)
		logger.info(f"[Engine] Loading embedding model '{embedding_model_name}'")
		self.embedding_generator = EmbeddingGenerator(embedding_model_name)  # embedder
		# Create ranker with default weights
		self.ranker = Ranker()  # ranker instance

		# Build parser with knowledge of known actors/directors for better classification
		actors = set()  # accumulator for all actors
		directors = set()  # accumulator for all directors
		for m in movies:  # iterate dataset
			actors.update(m.actors)  # add actors
			if m.director:  # if director present
				directors.add(m.director)  # add director
		self.parser = QueryParser(known_actors=actors, known_directors=directors)  # parser
		logger.info(f"[Engine] Parser ready with {len(actors)} actors and {len(directors)} directors")

		# Precompute popularity bounds for metadata normalization in ranking
		pops = [m.popularity for m in movies if m.popularity is not None]  # collect popularity
		self.pop_bounds = (min(pops) if pops else 0.0, max(pops) if pops else 1.0)  # min/max
		logger.debug(f"[Engine] Popularity bounds: {self.pop_bounds}")  # diagnostic

		# Initialize or load FAISS index depending on saved files
		if load_index_base_path and self._index_files_exist(load_index_base_path):  # saved index?
			logger.info(f"[Engine] Loading saved FAISS index from '{load_index_base_path}'")
			self.vector_store = VectorStore.load_index(load_index_base_path, movies=movies)  # load index
		else:
			logger.info("[Engine] Building FAISS index from scratch (no saved index found)")
			self.vector_store = VectorStore(
				embedding_dimension=self.embedding_generator.get_embedding_dimension(),  # dimension
				use_cosine_similarity=use_cosine_similarity,  # metric
			)
			self._build_index()  # compute embeddings and add to index

	def _index_files_exist(self, base_path: str) -> bool:
		"""Check if both FAISS index and metadata files exist for a given base path."""
		base = Path(base_path)  # wrap path
		return base.with_suffix('.index').exists() and base.with_suffix('.pkl').exists()  # both present

	def _build_index(self):
		"""Generate embeddings for all movies and add them to the FAISS index."""
		logger.info("[Engine] Generating embeddings for movies...")  # progress
		movie_embeddings = self.embedding_generator.generate_movie_embeddings(self.movies, batch_size=32, show_progress=True)  # embedding matrix
		logger.info("[Engine] Adding embeddings to FAISS index...")  # progress
		self.vector_store.add_movies(self.movies, movie_embeddings)  # populate index
		logger.info(f"[Engine] Index ready with {self.vector_store.size()} vectors")  # summary

	def parse_query(self, query: str) -> ParsedQuery:
		"""Parse the raw query into a structured representation."""
		logger.debug(f"[Engine] Parsing query: {query}")  # trace
		return self.parser.parse(query)  # delegate to parser

	def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
		"""Run semantic search, apply filters, and rank results."""
		parsed = self.parse_query(query)  # structured query
		logger.debug(
			"[Engine] Parsed summary | genres=%s actors=%s directors=%s year=%s keywords=%s",
			parsed.genres,
			parsed.actors[:5],
			parsed.directors[:5],
			parsed.year_range,
			parsed.keywords,
		)

		# Vector search to get candidates (retrieve more than top_k to allow filtering)
		logger.debug("[Engine] Generating query embedding and searching FAISS")  # trace
		q_emb = self.embedding_generator.generate_query_embedding(parsed.raw_query)  # embed query
		candidates = self.vector_store.search(q_emb, top_k=max(100, top_k))  # search many
		logger.debug(f"[Engine] Retrieved {len(candidates)} candidate vectors")  # count

		# Convert candidates to movies with similarity and compute final scores
		results: List[SearchResult] = []  # accumulator
		for movie_id, sim in candidates:  # each candidate
			movie = self.vector_store.get_movie_by_id(movie_id)  # fetch movie
			if not movie:  # safety check
				continue  # skip
			# Apply strict year filtering if specified in parsed query
			if parsed.year_range and movie.year:
				start, end = parsed.year_range  # unpack range
				if not (start <= movie.year <= end):  # outside range?
					logger.debug(
						"[Engine] Filtered out by year | movie=%s (%s) | movie_year=%s | required=%s-%s",
						movie.title,
						movie.id,
						movie.year,
						start,
						end,
					)
					continue  # reject

			# Compute final hybrid score
			score = self.ranker.compute_final_score(movie, parsed, semantic_similarity=sim, popularity_bounds=self.pop_bounds)  # score
			logger.debug(
				"[Engine] Candidate kept | movie=%s (%s) | sim=%.3f | final_score=%.3f",
				movie.title,
				movie.id,
				sim,
				score,
			)
			results.append(SearchResult(movie=movie, score=score, similarity=sim))  # collect

		# Sort results by final score descending and return top_k
		results.sort(key=lambda r: r.score, reverse=True)  # sort
		logger.info(f"[Engine] Returning top {min(top_k, len(results))} of {len(results)} ranked results")  # summary
		return results[:top_k]  # slice

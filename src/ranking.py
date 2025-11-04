"""
Ranking module.
Combines semantic similarity with filter matching and metadata signals to produce a final score.
"""

from typing import Tuple

from .models import Movie, ParsedQuery


class Ranker:
	"""
	Computes final ranking scores based on multiple signals:
	- semantic_similarity: cosine similarity from vector search (0..1)
	- filter matches: genres, actors, directors, year
	- metadata boosts: rating (0..10), popularity (arbitrary scale)
	"""

	def __init__(
		self,
		rating_weight: float = 0.05,
		popularity_weight: float = 0.05,
		semantic_weight: float = 0.6,
		match_weight: float = 0.3,
		min_year: int = 1900,
		max_year: int = 2100,
	):
		self.rating_weight = rating_weight
		self.popularity_weight = popularity_weight
		self.semantic_weight = semantic_weight
		self.match_weight = match_weight
		self.min_year = min_year
		self.max_year = max_year

	def compute_final_score(
		self,
		movie: Movie,
		parsed: ParsedQuery,
		semantic_similarity: float,
		popularity_bounds: Tuple[float, float],
	) -> float:
		"""
		Combine all signals into a single score (0..1).
		"""
		match_score = self._match_score(movie, parsed)
		metadata_score = self._metadata_score(movie, popularity_bounds)

		final_score = (
			self.semantic_weight * semantic_similarity +
			self.match_weight * match_score +
			self.rating_weight * metadata_score[0] +
			self.popularity_weight * metadata_score[1]
		)
		# Clamp
		return max(0.0, min(1.0, final_score))

	def _match_score(self, movie: Movie, parsed: ParsedQuery) -> float:
		"""
		Compute how well the movie matches the explicit filters in the query.
		Returns a score in [0..1].
		"""
		genre_score = 0.0
		actor_score = 0.0
		director_score = 0.0
		year_score = 0.0

		# Genres: proportion of query genres that appear in movie
		if parsed.genres:
			match_count = sum(1 for g in parsed.genres if g.lower() in [mg.lower() for mg in movie.genres])
			genre_score = match_count / max(1, len(parsed.genres))

		# Actors: proportion of query actors that appear in movie (case-insensitive exact)
		if parsed.actors:
			movie_actors = {a.lower() for a in movie.actors}
			match_count = sum(1 for a in parsed.actors if a.lower() in movie_actors)
			actor_score = match_count / max(1, len(parsed.actors))

		# Director: binary match
		if parsed.directors:
			movie_dir = (movie.director or '').lower()
			director_score = 1.0 if any(d.lower() == movie_dir for d in parsed.directors) else 0.0

		# Year range
		if parsed.year_range and movie.year:
			start, end = parsed.year_range
			year_score = 1.0 if (start <= movie.year <= end) else 0.0

		# Weighted average for matches
		# Give more weight to explicit people matches and year
		return (
			0.35 * genre_score +
			0.35 * actor_score +
			0.20 * director_score +
			0.10 * year_score
		)

	def _metadata_score(self, movie: Movie, popularity_bounds: Tuple[float, float]) -> Tuple[float, float]:
		"""
		Normalize rating and popularity to [0..1].
		Returns (rating_norm, popularity_norm)
		"""
		# Rating 0..10 to 0..1
		rating_norm = 0.0
		if movie.rating is not None:
			rating_norm = max(0.0, min(1.0, (movie.rating or 0.0) / 10.0))

		# Popularity normalization between min and max in dataset
		pop_min, pop_max = popularity_bounds
		popularity_norm = 0.0
		if pop_max > pop_min:
			popularity_norm = (max(pop_min, min(pop_max, (movie.popularity or 0.0))) - pop_min) / (pop_max - pop_min)

		return rating_norm, popularity_norm

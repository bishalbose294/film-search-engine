"""
Data loading and preprocessing module.
Handles loading movies from JSONL/CSV and cleaning/normalizing the data.
"""

# Standard libs for JSON parsing, regex, typing, and paths
import json  # read JSON lines
from typing import List, Dict  # type hints
from pathlib import Path  # filesystem-safe paths

# Import our Movie data class used across the project
from .models import Movie  # structured movie record

# Console logging
from loguru import logger  # console logger


class DataLoader:
	"""
	Handles loading and preprocessing of movie data.
	"""
	
	# Genre synonym mapping: common user phrasings → single standard name
	GENRE_SYNONYMS = {
		'sci-fi': 'Science Fiction',  # map hyphenated to canonical
		'sci fi': 'Science Fiction',  # map spaced form
			'sci fy': 'Science Fiction',  # common misspelling with y
		'science-fiction': 'Science Fiction',  # map with dash
		'science fiction': 'Science Fiction',  # map with space
		'scifi': 'Science Fiction',  # common variant
		'sci-fy': 'Science Fiction',  # typo variant
		'sci-fyi': 'Science Fiction',  # typo variant
		'horror': 'Horror',
		'thriller': 'Thriller',
		'comedy': 'Comedy',
		'funny': 'Comedy',
		'drama': 'Drama',
		'action': 'Action',
		'adventure': 'Adventure',
		'romance': 'Romance',
		'romantic': 'Romance',
		'fantasy': 'Fantasy',
		'mystery': 'Mystery',
		'crime': 'Crime',
		'war': 'War',
		'western': 'Western',
		'animation': 'Animation',
		'animated': 'Animation',
		'documentary': 'Documentary',
		'family': 'Family',
		'musical': 'Musical',
		'biography': 'Biography',
		'biographical': 'Biography',
		'sport': 'Sport',
		'sports': 'Sport',
	}
	
	def __init__(self):
		"""Initialize the data loader and expose the synonyms mapping."""
		self.genre_synonyms = self.GENRE_SYNONYMS  # store mapping for reuse
	
	def load_movies_from_jsonl(self, filepath: str) -> List[Movie]:
		"""
		Load movies from a JSON Lines (JSONL) file where each line is one JSON object.
		Returns a list of Movie objects.
		"""
		movies = []  # accumulator for parsed Movie objects
		filepath = Path(filepath)  # normalize path
		
		# Validate the file presence early to give clear error messages
		if not filepath.exists():
			raise FileNotFoundError(f"Movie data file not found: {filepath}")
		
		logger.info(f"[DataLoader] Loading movies from {filepath}...")  # log action
		
		# Open the file and read line-by-line to handle large datasets efficiently
		with open(filepath, 'r', encoding='utf-8') as f:
			for line_num, line in enumerate(f, 1):  # keep track of line number for diagnostics
				try:
					data = json.loads(line.strip())  # parse JSON object per line
					movie = self._parse_movie_data(data)  # convert dict -> Movie
					movies.append(movie)  # collect
				except json.JSONDecodeError as e:
					logger.warning(f"[DataLoader] Skipping invalid JSON at line {line_num}: {e}")  # malformed line
					continue  # move on
				except Exception as e:
					logger.warning(f"[DataLoader] Error parsing movie at line {line_num}: {e}")  # unexpected issue
					continue  # move on
		
		logger.info(f"[DataLoader] Successfully loaded {len(movies)} movies.")  # summary
		return movies  # return list
	
	def _parse_movie_data(self, data: Dict) -> Movie:
		"""
		Convert a raw dictionary (from file) into a strongly-typed Movie object.
		Performs normalization and safe defaults.
		"""
		# Parse fields that may arrive as comma-separated strings or lists
		genres = self._parse_comma_separated(data.get('genres', []))  # list of genres
		actors = self._parse_comma_separated(data.get('actors', []))  # list of actors
		characters = self._parse_comma_separated(data.get('characters', []))  # optional roles
		tags = self._parse_comma_separated(data.get('tags', []))  # optional tags
		
		# Normalize genres using our mapping so variants collapse to one canonical label
		normalized_genres = [self._normalize_genre(g) for g in genres]
		
		# Normalize free-text fields to improve search quality
		title = self._normalize_text(data.get('title', ''))  # clean title
		overview = self._normalize_text(data.get('overview', ''))  # clean description
		director = self._normalize_text(data.get('director', ''))  # clean director
		
		# Normalize actor names to a consistent lowercase, trimmed form
		normalized_actors = [self._normalize_text(actor) for actor in actors]
		
		# Parse numeric fields, providing safe defaults when missing
		year = int(data.get('year', 0)) if data.get('year') else 0  # int year or 0
		rating = float(data.get('rating', 0.0)) if data.get('rating') else 0.0  # float rating
		votes = int(data.get('votes', 0)) if data.get('votes') else 0  # int votes
		popularity = float(data.get('popularity', 0.0)) if data.get('popularity') else 0.0  # float popularity
		budget = float(data.get('budget', 0)) if data.get('budget') else None  # optional float budget
		
		# Assemble the Movie object with normalized and raw values
		movie = Movie(
			id=str(data.get('id', '')),  # ensure ID is string
			title=title,  # normalized title
			overview=overview,  # normalized overview
			genres=normalized_genres,  # canonical genres
			director=director,  # normalized director
			actors=normalized_actors,  # normalized actors
			characters=characters if characters else None,  # optional
			year=year,  # numeric
			rating=rating,  # numeric
			votes=votes,  # numeric
			popularity=popularity,  # numeric
			budget=budget if budget else None,  # optional
			poster_url=data.get('url') or data.get('poster_url'),  # prefer 'url' then 'poster_url'
			tags=tags if tags else None  # optional list
		)
		
		# Build a single weighted text string used by the embedding model later
		movie.searchable_text = self._create_searchable_text(movie)
		
		return movie  # return structured movie
	
	def _parse_comma_separated(self, value) -> List[str]:
		"""
		Normalize a value that may be None, a list, or a comma-separated string
		into a list of clean strings.
		"""
		if value is None:  # missing field
			return []  # treat as empty list
		if isinstance(value, list):  # already a list
			return [str(item).strip() for item in value if item]  # clean each
		if isinstance(value, str):  # comma-separated string
			return [item.strip() for item in value.split(',') if item.strip()]  # split/trim
		return []  # any other type becomes empty
	
	def _normalize_text(self, text: str) -> str:
		"""
		Lowercase and trim whitespace; handle None safely by returning empty string.
		"""
		if not text:  # None or empty
			return ''  # normalize to empty
		return text.strip().lower()  # standardized text
	
	def _normalize_genre(self, genre: str) -> str:
		"""
		Map a raw genre to its canonical form using synonyms; fall back to Title Case.
		"""
		if not genre:  # missing genre
			return ''
		
		genre_lower = genre.strip().lower()  # prepare for lookup
		
		# If present in synonyms, return canonical value
		if genre_lower in self.genre_synonyms:
			return self.genre_synonyms[genre_lower]
		
		# Otherwise title-case the input to standardize (e.g., "war drama" → "War Drama")
		return genre.strip().title()
	
	def _create_searchable_text(self, movie: Movie) -> str:
		"""
		Create a weighted combined text that emphasizes the most important fields.
		Weights: title (3x), genres (2x), director (2x), actors (2x), overview (1x).
		"""
		parts = []  # accumulate text segments
		
		# Title weighted most because users often search by it
		if movie.title:
			parts.extend([movie.title] * 3)
		
		# Repeat each genre to give it moderate influence
		for genre in movie.genres:
			parts.extend([genre] * 2)
		
		# Director name also helps in matching creator searches
		if movie.director:
			parts.extend([movie.director] * 2)
		
		# Actors often appear in people-based searches
		for actor in movie.actors:
			parts.extend([actor] * 2)
		
		# Overview gives broader context and keywords
		if movie.overview:
			parts.append(movie.overview)
		
		# Join all pieces into one string to feed the embedding model
		return ' '.join(parts)
	
	def get_all_actors(self, movies: List[Movie]) -> List[str]:
		"""Return a sorted list of all unique actor names in the dataset."""
		actors = set()  # collect unique actors
		for movie in movies:  # iterate dataset
			actors.update(movie.actors)  # add movie actors
		return sorted(list(actors))  # sorted for stable display
	
	def get_all_directors(self, movies: List[Movie]) -> List[str]:
		"""Return a sorted list of all unique director names in the dataset."""
		directors = set()  # unique directors
		for movie in movies:  # iterate
			if movie.director:  # ignore empty
				directors.add(movie.director)
		return sorted(list(directors))  # sorted output
	
	def get_all_genres(self, movies: List[Movie]) -> List[str]:
		"""Return a sorted list of all unique genres in the dataset."""
		genres = set()  # unique genres
		for movie in movies:  # iterate
			genres.update(movie.genres)
		return sorted(list(genres))  # sorted output


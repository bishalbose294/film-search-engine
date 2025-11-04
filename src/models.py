"""
Data models for the Film Search Engine.
Defines the core data structures used throughout the system.
"""

# Import dataclass to define simple "record-like" classes without boilerplate
from dataclasses import dataclass  # auto-generates __init__, __repr__, etc.
# Import typing helpers for precise and self-documenting types
from typing import List, Optional, Tuple  # lists, optional values, and fixed-size tuples


@dataclass
class Movie:
	"""
	Represents a single movie and all the information we know about it.
	These fields are used both for searching and for showing data in results.
	"""
	id: str  # unique identifier of the movie (string for consistency)
	title: str  # movie title in lowercase (normalized)
	overview: str  # short description/synopsis in lowercase (normalized)
	genres: List[str]  # list of genre names (e.g., ["Drama", "War"]) in canonical form
	director: str  # director's name in lowercase (normalized)
	actors: List[str]  # list of actor names in lowercase (normalized)
	year: int  # release year as a number (e.g., 1999)
	rating: float  # average user/critic rating on a 0-10 scale
	votes: int  # number of votes/ratings the movie has received
	popularity: float  # popularity score from dataset (used for gentle ranking boost)
	characters: Optional[List[str]] = None  # optional: character names associated with actors
	budget: Optional[float] = None  # optional: movie budget if known
	poster_url: Optional[str] = None  # optional: URL for the poster image (for UI)
	tags: Optional[List[str]] = None  # optional: free-form tags from the dataset
	searchable_text: Optional[str] = None  # generated text used to create embeddings


@dataclass
class ParsedQuery:
	"""
	Represents the meaning we extract from the user's natural-language query.
	We separate what the user wants (genres, years, people) from the raw text.
	"""
	raw_query: str  # the original text the user typed
	genres: List[str]  # normalized genre names identified in the query
	actors: List[str]  # actor names found or inferred from the query
	directors: List[str]  # director names found or inferred from the query
	year_range: Optional[Tuple[int, int]]  # e.g., (1990, 1999) for "90s" queries
	keywords: List[str]  # leftover important words (context) after extracting entities


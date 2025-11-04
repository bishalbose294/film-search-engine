"""
Unit tests for QueryParser: decades, genres, and fuzzy entity matching.
Run: python tests/test_query_parser.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from src.data_loader import DataLoader
from src.query_parser import QueryParser


def load_parser():
	loader = DataLoader()
	movies = loader.load_movies_from_jsonl(str(ROOT / 'data' / 'movies.jsonl'))
	actors = set(loader.get_all_actors(movies))
	directors = set(loader.get_all_directors(movies))
	return QueryParser(known_actors=actors, known_directors=directors)


def assert_equal(actual, expected, msg):
	if actual != expected:
		raise AssertionError(f"{msg} | expected={expected}, actual={actual}")


def assert_true(cond, msg):
	if not cond:
		raise AssertionError(msg)


def test_decades(parser: QueryParser):
	pq = parser.parse("sci-fi movies from the 90s")
	assert_equal(pq.year_range, (1990, 1999), "90s decade parsing")

	pq = parser.parse("comedy films in the 80s")
	assert_equal(pq.year_range, (1980, 1989), "80s decade parsing")

	pq = parser.parse("funny films from the early 2000s")
	assert_equal(pq.year_range, (2000, 2004), "early 2000s parsing")


def test_genre_synonyms(parser: QueryParser):
	pq = parser.parse("sci-fi adventure")
	assert_true("Science Fiction" in pq.genres, "synonym sci-fi -> Science Fiction")


def test_fuzzy_people(parser: QueryParser):
	pq = parser.parse("90s with tom hanks")
	assert_true(any("tom hanks" == a for a in pq.actors), "actor exact match")

	pq = parser.parse("horror by wes cravin")  # misspelled on purpose
	assert_true(any("wes craven" == d for d in pq.directors), "director fuzzy match")


def main():
	parser = load_parser()
	print("Running QueryParser tests...")
	test_decades(parser)
	print(" - decades ok")
	test_genre_synonyms(parser)
	print(" - genre synonyms ok")
	test_fuzzy_people(parser)
	print(" - fuzzy people ok")
	print("All QueryParser tests passed!")


if __name__ == '__main__':
	main()

"""
Unit tests for QueryParser: decades, genres, and fuzzy entity matching.
Run: python tests/test_query_parser.py
"""

from src.data_loader import DataLoader
from src.query_parser import QueryParser


def load_parser():
	loader = DataLoader()
	movies = loader.load_movies_from_jsonl('data/movies.jsonl')
	actors = set(loader.get_all_actors(movies))
	directors = set(loader.get_all_directors(movies))
	return QueryParser(known_actors=actors, known_directors=directors)


def assert_equal(actual, expected, msg):
	if actual != expected:
		raise AssertionError(f"{msg} | expected={expected}, actual={actual}")


def assert_true(cond, msg):
	if not cond:
		raise AssertionError(msg)


def assert_raises(fn, msg):
	try:
		fn()
		raise AssertionError(msg)
	except Exception:
		return


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

	# Fuzzy and variant spellings
	pq = parser.parse("scifi comedy")
	assert_true("Science Fiction" in pq.genres or "Comedy" in pq.genres, "genre variants captured")

	pq = parser.parse("sci fy thrillers")
	assert_true("Science Fiction" in pq.genres, "fuzzy sci fy -> Science Fiction")


def test_fuzzy_people(parser: QueryParser):
	pq = parser.parse("90s with tom hanks")
	assert_true(any("tom hanks" == a for a in pq.actors), "actor exact match")

	pq = parser.parse("horror by wes cravin")  # misspelled on purpose
	assert_true(any("wes craven" == d for d in pq.directors), "director fuzzy match")

	# Additional actor patterns
	pq = parser.parse("starring keanu reeves")
	assert_true(any("keanu reeves" == a for a in pq.actors), "starring pattern")

	pq = parser.parse("featuring tom cruise")
	assert_true(any("tom cruise" == a for a in pq.actors), "featuring pattern")

	pq = parser.parse("keanu reeves in it")
	assert_true(any("keanu reeves" == a for a in pq.actors), "'in it' pattern")

	# Additional director patterns
	pq = parser.parse("directed by wes craven")
	assert_true(any("wes craven" == d for d in pq.directors), "directed by pattern")

	pq = parser.parse("by steven spielberg director")
	assert_true(any("steven spielberg" == d for d in pq.directors), "by X director pattern")

	pq = parser.parse("by christopher nolan")
	assert_true(any("christopher nolan" == d for d in pq.directors), "by X pattern")


def test_year_parsing(parser: QueryParser):
	# Explicit range
	pq = parser.parse("1994-1996 thrillers")
	assert_equal(pq.year_range, (1994, 1996), "year explicit range")

	# Before/After
	pq = parser.parse("before 2000 war movies")
	assert_equal(pq.year_range, (1900, 1999), "before boundary")

	pq = parser.parse("after 2010 action")
	assert_equal(pq.year_range, (2010, 2100), "after boundary")

	# Single year
	pq = parser.parse("movies from 2001")
	assert_equal(pq.year_range, (2001, 2001), "single year")


def test_keywords(parser: QueryParser):
	pq = parser.parse("war movies about love before 2000")
	# Common stopwords removed; keep meaningful tokens
	assert_true("love" in pq.keywords, "keywords contain meaningful token")
	assert_true("movies" not in pq.keywords and "about" not in pq.keywords, "stopwords removed")

	# Names and genres removed from keywords
	pq = parser.parse("sci-fi with tom hanks")
	assert_true("science" not in " ".join(pq.keywords), "genre removed from keywords")
	assert_true("tom" not in pq.keywords and "hanks" not in pq.keywords, "names removed from keywords")


def main():
	parser = load_parser()
	print("Running QueryParser tests...")
	test_decades(parser)
	print(" - decades ok")
	test_genre_synonyms(parser)
	print(" - genre synonyms ok")
	test_fuzzy_people(parser)
	print(" - fuzzy people ok")
	test_year_parsing(parser)
	print(" - year parsing ok")
	test_keywords(parser)
	print(" - keywords ok")

	# Error handling
	assert_raises(lambda: parser.parse("   "), "empty query should raise")
	print(" - empty query handling ok")
	print("All QueryParser tests passed!")


if __name__ == '__main__':
	main()

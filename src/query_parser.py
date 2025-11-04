"""
Query parsing module.
Extracts entities (genres, actors, directors), date ranges, and keywords from natural language queries.
Falls back gracefully if spaCy model is unavailable.
"""

import re  # regex for date/keyword extraction
from typing import List, Optional, Tuple, Set  # type annotations

try:
	import spacy  # NLP pipeline for named entities
	_SPACY_AVAILABLE = True  # flag if spaCy is importable
except Exception:
	_SPACY_AVAILABLE = False  # fallback path when spaCy unavailable

from rapidfuzz import process, fuzz  # fuzzy matching utilities

from loguru import logger  # console logging

from .models import ParsedQuery  # structured query representation
from .data_loader import DataLoader  # for genre synonyms


class QueryParser:
	"""
	Parses natural language queries into a structured ParsedQuery.
	Uses spaCy NER to find PERSON entities (actors/directors), regex for date ranges,
	and a synonym map for genres with fuzzy matching support.
	"""

	# Pre-compiled regex patterns for date expressions
	RE_DECADE = re.compile(r"(?P<prefix>early|mid|late)?\s*(?P<decade>\d{2})\s*'?s", re.I)  # '90s, mid 80s
	RE_CENTURY_DECADE = re.compile(r"(?P<prefix>early|mid|late)?\s*(?P<century>\d{4})\s*'?s", re.I)  # early 2000s
	RE_YEAR = re.compile(r"\b(19\d{2}|20\d{2})\b")  # single year like 1995
	RE_RANGE = re.compile(r"\b(19\d{2}|20\d{2})\s*[-–to]{1,3}\s*(19\d{2}|20\d{2})\b", re.I)  # 1990-1999
	RE_BEFORE = re.compile(r"before\s+(19\d{2}|20\d{2})", re.I)  # before 2000
	RE_AFTER = re.compile(r"after\s+(19\d{2}|20\d{2})", re.I)  # after 2010

	# Genre keywords (normalized) for extraction beyond synonyms
	GENRE_KEYWORDS: Set[str] = set([g.lower() for g in DataLoader.GENRE_SYNONYMS.values()])  # canonical genre names
	GENRE_SYNONYMS = {k.lower(): v for k, v in DataLoader.GENRE_SYNONYMS.items()}  # mapping variants->canonical

	def __init__(self, known_actors: Optional[Set[str]] = None, known_directors: Optional[Set[str]] = None):
		# Lowercased known names for efficient membership and fuzzy matching reference
		self.known_actors = {a.lower() for a in (known_actors or set())}
		self.known_directors = {d.lower() for d in (known_directors or set())}

		# Initialize spaCy model if available; otherwise proceed without it
		self._nlp = None
		if _SPACY_AVAILABLE:
			try:
				self._nlp = spacy.load("en_core_web_sm")
				logger.debug("[Parser] spaCy model en_core_web_sm loaded")
			except Exception as e:
				logger.warning(f"[Parser] spaCy load failed, falling back to heuristics: {e}")
				self._nlp = None

		# Pre-build lists for fuzzy search to avoid recreating on each parse
		self._actor_list = list(self.known_actors)
		self._director_list = list(self.known_directors)
		self._genre_list = sorted(list(self.GENRE_KEYWORDS))
		logger.debug(f"[Parser] Initialized with {len(self._actor_list)} actors, {len(self._director_list)} directors")

	def parse(self, query: str) -> ParsedQuery:
		"""Main entry: produce a ParsedQuery from a raw string."""
		if not query or not query.strip():  # empty input guard
			raise ValueError("Query cannot be empty")

		q = query.strip().lower()  # normalize spaces and casing
		logger.debug(f"[Parser] Input query: '{query}' -> normalized: '{q}'")  # trace

		# 1) Extract year range
		year_range = self._extract_year_range(q)

		# 2) Extract genres (synonyms + direct keywords + fuzzy)
		genres = self._extract_genres(q)

		# 3) Extract people (actors/directors) with fuzzy
		actors, directors = self._extract_people(q)

		# 4) Remaining keywords (rough heuristic): remove known tokens
		keywords = self._extract_keywords(q, genres, actors, directors)

		# Package into structured object
		parsed = ParsedQuery(
			raw_query=query,
			genres=sorted(list(genres)),
			actors=sorted(list(actors)),
			directors=sorted(list(directors)),
			year_range=year_range,
			keywords=keywords
		)
		logger.debug(
			"[Parser] Parsed result | genres=%s | actors=%s | directors=%s | year_range=%s | keywords=%s",
			parsed.genres,
			parsed.actors[:5],
			parsed.directors[:5],
			parsed.year_range,
			parsed.keywords,
		)
		return parsed

	def _extract_year_range(self, q: str) -> Optional[Tuple[int, int]]:
		logger.debug("[Parser] Year extraction: scanning '%s'", q)
		# 1990-1999 explicit range
		r = self.RE_RANGE.search(q)
		if r:
			start, end = int(r.group(1)), int(r.group(2))
			logger.debug("[Parser] Found explicit range match: %s-%s", start, end)
			if start > end:  # normalize order
				start, end = end, start
			logger.debug("[Parser] Normalized range -> (%s, %s)", start, end)
			return (start, end)

		# before / after boundaries
		m = self.RE_BEFORE.search(q)
		if m:
			end = int(m.group(1))
			res = (1900, end - 1)
			logger.debug("[Parser] Found 'before' boundary: < %s -> range %s", end, res)
			return res
		m = self.RE_AFTER.search(q)
		if m:
			start = int(m.group(1))
			res = (start, 2100)
			logger.debug("[Parser] Found 'after' boundary: > %s -> range %s", start, res)
			return res

		# early/mid/late 2000s
		m = self.RE_CENTURY_DECADE.search(q)
		if m:
			prefix = (m.group("prefix") or "").lower()
			century = int(m.group("century"))
			res = self._prefix_to_range(century, prefix)
			logger.debug("[Parser] Found century-decade '%s' with prefix '%s' -> %s", century, prefix or "-", res)
			return res

		# 90s/80s decade mapping (00-29->2000s, 30-99->1900s)
		m = self.RE_DECADE.search(q)
		if m:
			prefix = (m.group("prefix") or "").lower()
			dec = int(m.group("decade"))
			base = 1900 if dec >= 30 else 2000
			res = self._prefix_to_range(base + dec, prefix)
			logger.debug("[Parser] Found decade '%s' (base %s) prefix '%s' -> %s", dec, base, prefix or "-", res)
			return res

		# single year
		m = self.RE_YEAR.search(q)
		if m:
			y = int(m.group(0))
			logger.debug("[Parser] Found single year -> (%s, %s)", y, y)
			return (y, y)

		logger.debug("[Parser] No year information found")
		return None  # nothing found

	def _prefix_to_range(self, decade_start: int, prefix: str) -> Tuple[int, int]:
		# Convert optional prefix into a sub-range within the decade
		if prefix == "early":  # first half-decade
			return (decade_start, decade_start + 4)
		if prefix == "mid":  # second half-decade
			return (decade_start + 5, decade_start + 9)
		if prefix == "late":  # last third bias
			return (decade_start + 7, decade_start + 9)
		return (decade_start, decade_start + 9)  # full decade default

	def _extract_genres(self, q: str) -> Set[str]:
		logger.debug("[Parser] Genre extraction from: '%s'", q)
		genres: Set[str] = set()  # accumulator
		# Synonym substring matches (e.g., "sci-fi" -> "Science Fiction")
		for key, val in self.GENRE_SYNONYMS.items():
			if key in q:
				genres.add(val)
				logger.debug("[Parser] Genre synonym match: '%s' -> '%s'", key, val)
		# Direct keyword hits of canonical names
		for g in self.GENRE_KEYWORDS:
			if g in q:
				genres.add(g.title())
				logger.debug("[Parser] Genre keyword match: '%s'", g)
		# Fuzzy token-level match to handle small typos/variants
		for token in set(re.findall(r"[a-z]+", q)):
			match, score, _ = process.extractOne(token, self._genre_list, scorer=fuzz.ratio)
			if match and score >= 88:
				genres.add(match.title())
				logger.debug("[Parser] Genre fuzzy match: token='%s' -> '%s' (score=%s)", token, match, score)
		return genres

	def _extract_people(self, q: str) -> Tuple[Set[str], Set[str]]:
		logger.debug("[Parser] People extraction from: '%s'", q)
		actors: Set[str] = set()  # matched actors
		directors: Set[str] = set()  # matched directors

		candidates: Set[str] = set()  # raw name candidates

		# Rule-based extraction for common patterns like "with tom hanks", "starring keanu reeves"
		for m in re.finditer(r"\b(with|starring|featuring)\s+([a-z]+\s+[a-z]+)\b", q):
			name = m.group(2).strip().lower()
			candidates.add(name)
			logger.debug("[Parser] Pattern people match: '%s' -> '%s'", m.group(1), name)

		# Rule-based extraction for director phrases: "directed by wes cravin", "by wes cravin"
		director_cands: Set[str] = set()
		for m in re.finditer(r"\bdirected\s+by\s+([a-z]+\s+[a-z]+)\b", q):
			name = m.group(1).strip().lower()
			director_cands.add(name)
			logger.debug("[Parser] Pattern director match: 'directed by' -> '%s'", name)
		for m in re.finditer(r"\bby\s+([a-z]+\s+[a-z]+)\b", q):
			name = m.group(1).strip().lower()
			director_cands.add(name)
			logger.debug("[Parser] Pattern director match: 'by' -> '%s'", name)

		# Use spaCy NER when available for PERSON entities
		if self._nlp is not None:
			doc = self._nlp(q)
			for ent in doc.ents:
				if ent.label_ == "PERSON":
					candidates.add(ent.text.strip().lower())
			logger.debug("[Parser] spaCy PERSON candidates: %s", list(candidates)[:10])
		else:
			# Fallback heuristic: consider bigrams as potential names
			for a, b in zip(q.split(), q.split()[1:]):
				candidates.add(f"{a} {b}")
			logger.debug("[Parser] Heuristic name candidates: %s", list(candidates)[:10])

		# Preferential fuzzy match for explicit director candidates
		for cand in director_cands:
			best_director = process.extractOne(cand, self._director_list, scorer=fuzz.WRatio)
			if best_director and best_director[1] >= 80:
				directors.add(best_director[0])
				logger.debug("[Parser] Director fuzzy match (pattern): '%s' -> '%s' (score=%s)", cand, best_director[0], best_director[1])

		# General fuzzy match candidates against known lists
		for cand in candidates:
			best_actor = process.extractOne(cand, self._actor_list, scorer=fuzz.WRatio)
			best_director = process.extractOne(cand, self._director_list, scorer=fuzz.WRatio)
			if best_actor and best_actor[1] >= 85:
				actors.add(best_actor[0])
				logger.debug("[Parser] Actor fuzzy match: '%s' -> '%s' (score=%s)", cand, best_actor[0], best_actor[1])
			elif best_director and best_director[1] >= 85:
				directors.add(best_director[0])
				logger.debug("[Parser] Director fuzzy match: '%s' -> '%s' (score=%s)", cand, best_director[0], best_director[1])

		# Bonus: add exact substrings, which helps when the full name appears verbatim
		for name in self._actor_list[:10000]:  # cap for performance
			if name in q:
				actors.add(name)
				logger.debug("[Parser] Actor substring match: '%s'", name)
		for name in self._director_list[:10000]:
			if name in q:
				directors.add(name)
				logger.debug("[Parser] Director substring match: '%s'", name)

		# Conservative fallback: if no actor found from known lists, use plausible two-word PERSON candidates
		if not actors and candidates:
			stop = {"with", "by", "from", "the", "in", "of", "and", "or"}
			plausible = set()
			for cand in candidates:
				parts = cand.split()
				if len(parts) == 2 and all(p.isalpha() for p in parts) and all(p not in stop for p in parts):
					plausible.add(cand)
			if plausible:
				logger.debug("[Parser] Fallback: adding plausible actor candidates %s", list(plausible))
				actors.update(plausible)

		return actors, directors

	def _extract_keywords(self, q: str, genres: Set[str], actors: Set[str], directors: Set[str]) -> List[str]:
		logger.debug("[Parser] Keyword extraction from: '%s'", q)
		# Tokenize to alphanumerics/apostrophes
		tokens = re.findall(r"[a-z0-9']+", q)
		# Build removal set: genres, synonyms, matched names, and common stop terms
		remove: Set[str] = set()
		remove.update([g.lower() for g in genres])
		remove.update(self.GENRE_SYNONYMS.keys())
		remove.update([a for a in actors])
		remove.update([d for d in directors])
		remove.update({
			"movie", "movies", "film", "films", "starring", "with", "featuring",
			"directed", "by", "from", "in", "the", "of", "about", "and", "or"
		})
		# Keep tokens not in removal set, deduplicate while preserving order
		keywords = [t for t in tokens if t not in remove]
		logger.debug("[Parser] Tokens: %s", tokens)
		logger.debug("[Parser] Removed tokens: %s", sorted(list(remove))[:50])
		seen = set()
		uniq = []
		for t in keywords:
			if t not in seen:
				seen.add(t)
				uniq.append(t)
		logger.debug("[Parser] Final keywords: %s", uniq)
		return uniq

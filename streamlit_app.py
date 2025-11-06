"""
Streamlit UI for Film Search Engine.
Calls the local FastAPI server at http://localhost:8000 to fetch search results,
or runs locally by loading the saved FAISS index (models/faiss_index.*) like the API does.

Run API (optional):   uvicorn api:app --reload
Run UI:                streamlit run streamlit_app.py
"""

# HTTP client to call the API when running in API mode
import requests  # make web requests to the FastAPI server
# Streamlit framework to build a simple interactive UI
import streamlit as st  # UI primitives
# Path utilities to find saved index files
from pathlib import Path  # path handling
# Typing to make function signatures clearer
from typing import Optional  # indicates values can be None

# Local engine imports for fallback/local mode (when API isn't used)
from src.data_loader import DataLoader  # load movies from file
from src.search_engine import SearchEngine  # run parsing + search + ranking

# Default URL where the FastAPI server is expected to run locally
DEFAULT_API_URL = "http://localhost:8000"  # default API base URL

# Configure Streamlit page (title and layout width)
st.set_page_config(page_title="Film Search Engine", layout="wide")  # wide layout

# Main page title
st.title("🎬 CineMind – A Smart Movie Search Engine")  # friendly header

# Cache the local engine so we only load the model/index once per session
@st.cache_resource(show_spinner=True)
def init_local_engine() -> Optional[SearchEngine]:
	"""Create a local SearchEngine and load saved FAISS index if available."""
	try:
		loader = DataLoader()  # create loader
		movies = loader.load_movies_from_jsonl('data/movies.jsonl')  # read dataset
		index_base = Path('models') / 'faiss_index'  # saved index base path
		# Engine will validate/load or build/save as needed
		engine = SearchEngine(movies, load_index_base_path=str(index_base))
		return engine  # success
	except Exception as e:
		# Show an error in the UI so users know local mode failed
		st.error(f"Failed to initialize local search engine: {e}")
		return None  # signal failure

# Sidebar contains configuration controls
with st.sidebar:
	st.header("Settings")  # section label
	top_k = st.slider("Top K Results", min_value=5, max_value=20, value=10)  # number of results to show
	api_url = st.text_input("API URL", DEFAULT_API_URL)  # where the API lives
	# Toggle to force local mode; if API health probe fails we also fall back to local
	use_local = st.toggle("Use local engine (auto-load saved index)", value=False, help="If enabled or API is unreachable, the app will run fully locally.")

# If not forcing local, check quickly whether the API is reachable
api_available = False  # default assumption
if not use_local:
	try:
		h = requests.get(f"{api_url}/health", timeout=3)  # ping API health endpoint
		api_available = h.ok  # True if server responded 200 OK
	except Exception:
		api_available = False  # probe failed
		st.sidebar.info("API not reachable; will use local engine.")  # inform user

# Initialize local engine only when needed (user toggle or API not available)
local_engine: Optional[SearchEngine] = None  # placeholder
if use_local or not api_available:
	with st.spinner("Initializing local engine (loading saved index if present)..."):
		local_engine = init_local_engine()  # build or load engine
		if local_engine is not None:
			st.sidebar.success("Local engine ready.")  # success note
		else:
			st.sidebar.error("Local engine failed to initialize.")  # error note

# Main text input where users type a natural language query
query = st.text_input("Enter your movie query", placeholder="e.g., sci-fi movies from the 90s with Tom Hanks")

# Two-column layout: button on left, tips on right
col1, col2 = st.columns([1, 3])  # grid with ratio 1:3
with col1:
	search_btn = st.button("Search", type="primary")  # triggers a search
with col2:
	st.caption("Tip: Try 'funny films from the early 2000s' or 'robert downey movies'")  # helpful examples

# When user clicks Search and the field isn't empty, perform the query
if search_btn and query.strip():
	with st.spinner("Searching..."):
		try:
			results_payload = None  # we will fill this with a uniform structure
			if local_engine is not None:
				# Local mode: run the full pipeline inside this process
				res = local_engine.search(query, top_k=top_k)  # get ranked results
				items = []  # convert to a dict similar to API output
				for r in res:
					m = r.movie  # movie object
					items.append({
						"movie": {
							"id": m.id,
							"title": m.title.title(),
							"year": m.year,
							"genres": m.genres,
							"director": m.director,
							"actors": m.actors[:5],
							"rating": m.rating,
							"popularity": m.popularity,
							"poster_url": m.poster_url,
							"overview": (m.overview[:350] if m.overview else None),
						},
						"score": round(r.score, 3),
						"similarity": round(r.similarity, 3)
					})
				results_payload = {
					"results": items,  # list of result entries
					"elapsed_ms": 0.0  # we skip timing for local UI simplicity
				}
			else:
				# API mode: call the server and let it perform the search
				resp = requests.get(f"{api_url}/search", params={"q": query, "top_k": top_k}, timeout=60)
				resp.raise_for_status()  # raise error if server responded with an error code
				results_payload = resp.json()  # parse JSON returned by API

			# Report success and how long it took (if API provided timing)
			st.success(f"Found {len(results_payload.get('results', []))} results in {results_payload.get('elapsed_ms', 0)} ms")
			st.divider()  # visual separator

			# Render each result as an image + details row
			for i, item in enumerate(results_payload.get('results', []), start=1):
				movie = item['movie']  # movie details
				score = item['score']  # final score
				sim = item['similarity']  # similarity score

				c1, c2 = st.columns([1, 4])  # small image column + large text column
				with c1:
					if movie.get('poster_url'):
						st.image(movie['poster_url'], width='stretch')  # poster
				with c2:
					st.subheader(f"{i}. {movie['title']} ({movie['year']})")  # title + year
					st.caption(f"Score: {score:.3f} | Similarity: {sim:.3f}")  # scores
					st.write(f"Genres: {', '.join(movie['genres'])}")  # genres
					if movie.get('director'):
						st.write(f"Director: {movie['director'].title()}")  # director
					st.write(f"Actors: {', '.join([a.title() for a in movie['actors']])}")  # actors
					if movie.get('overview'):
						st.write(movie['overview'])  # synopsis
				st.divider()  # separator

		except requests.RequestException as e:  # network/API errors
			st.error(f"API request failed: {e}")  # show human-friendly message
		except Exception as e:  # any other runtime error
			st.error(f"Search failed: {e}")  # show error

# Show a footer indicator of current mode
st.sidebar.markdown("---")  # separator
if local_engine is not None:
	st.sidebar.caption("Mode: Local engine (saved index loaded if present)")  # mode label
else:
	st.sidebar.caption("Mode: API client (ensure uvicorn api:app --reload is running)")  # mode label

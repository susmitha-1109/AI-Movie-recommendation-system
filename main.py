from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import difflib
import os # Added for path handling

# --- Configuration & Initialization ---
app = FastAPI()

# 1. FIX: Correctly applying CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Loading and Preprocessing ---

# Assuming 'movies.csv' is in the root directory where the script is run
# You might need to adjust the path if the file is elsewhere.
# Current Path: "/content/movies.csv" - If you are running this in a specific environment like Colab.
# For local execution, you might just use: movie_data = pd.read_csv("movies.csv", ...)
# Using the original path from your code:
try:
    movie_data = pd.read_csv("/content/movies.csv", on_bad_lines='skip', engine='python')
except FileNotFoundError:
    # Fallback/Suggestion for local testing if the path is wrong
    print("WARNING: Could not find /content/movies.csv. Trying ./movies.csv")
    try:
        movie_data = pd.read_csv("movies.csv", on_bad_lines='skip', engine='python')
    except Exception as e:
        print(f"FATAL: Could not load movie data: {e}")
        # Exit or raise error if data can't be loaded
        raise

# Ensure 'index' column exists and is unique for mapping (important if it was missing)
if 'index' not in movie_data.columns:
    movie_data['index'] = movie_data.index

selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']

for feature in selected_features:
    # Replacing missing values with empty string for vectorization
    movie_data[feature] = movie_data[feature].fillna('')

# Combining the selected features into a single string
combined_features = (
    movie_data['genres'] + ' ' +
    movie_data['keywords'] + ' ' +
    movie_data['tagline'] + ' ' +
    movie_data['cast'] + ' ' +
    movie_data['director']
)

# --- TF-IDF Vectorization and Cosine Similarity ---

vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# Compute cosine similarity
similarity = cosine_similarity(feature_vectors)
#

# List of all movie titles for matching
list_of_all_titles = movie_data['title'].to_list()

# --- Pydantic Model for Request Body ---

class MovieRequest(BaseModel):
    movie: str

# --- FastAPI Routes ---

@app.get("/")
def root():
    return {"message": "Movie Recommendation API. Use /recommendations POST endpoint."}

@app.post("/recommendations")
def recommend(request: MovieRequest):
    movie_name = request.movie.strip() # Strip whitespace

    # Use difflib to find close matches. It returns a list.
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

    if not find_close_match:
        return {
            "error": "Movie not found",
            "message": f"Could not find a close match for '{movie_name}'. Please check the spelling."
        }

    # 2. FIX: Select the best match (the first one)
    close_match = find_close_match[0]

    # Find the index of the best match in the original DataFrame
    # Note: .iloc[0] is used to safely get the value from the Series/DataFrame row
    index_of_the_movie = movie_data[movie_data.title == close_match]['index'].iloc[0]

    # Get the similarity scores for the matched movie
    similarity_score = list(enumerate(similarity[index_of_the_movie]))

    # Sort the movies based on the similarity score in descending order
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    recommendations = []

    # Iterate through the top 30 similar movies (skipping the first one, which is the input movie itself)
    # 3. FIX: Move the 'return' statement outside the loop
    for i, movie in enumerate(sorted_similar_movies[1:30]):
        index = movie[0]
        # Get the title using the index
        title = movie_data[movie_data.index == index]['title'].values[0]
        recommendations.append(title)

    # Return the result after the loop is complete
    return {
        "matched_movie": close_match,
        "recommendations": recommendations
    }

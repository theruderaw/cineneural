from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import schemas
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import re

app = FastAPI(title="Movie Recommendation System")

# Mount the static directory for CSS/JS
app.mount("/static", StaticFiles(directory="static"), name="static")

def word_tokenizer(text):
    return re.findall(r'\w+', str(text).lower())

print("Loading Multi-Dimensional TF-IDF Models into RAM...")
try:
    df_slim    = pd.read_pickle("models/dataset_metadata.pkl")
    mat_genre   = joblib.load("models/matrix_genre.pkl")
    mat_origin  = joblib.load("models/matrix_origin.pkl")
    mat_cast    = joblib.load("models/matrix_cast.pkl")
    mat_plot    = joblib.load("models/matrix_plot.pkl")
    tfidf_plot  = joblib.load("models/tfidf_plot.pkl")
    print("Models loaded successfully!")
except Exception as e:
    print(f"Warning: Models not found ({e}). Run train_model.py first!")

# --- Concept-Aware Layer (Symbolic AI Anchors) ---
CONCEPT_BUCKETS = {
    "Celestial/Cosmic": ["space", "astronaut", "planet", "galaxy", "black hole", "relativity", "star", "nasa", "orbit", "spacecraft", "celestial", "cosmic", "wormhole"],
    "Military/Conflict": ["military", "war", "soldier", "battle", "army", "combat", "weapon", "officer", "base", "defense", "sergeant", "colonel", "command"],
    "Dystopian/System": ["government", "authority", "control", "system", "power", "dystopian", "rebellion", "police", "surveillance", "corporation"],
    "Family/Emotional": ["family", "daughter", "son", "father", "mother", "emotional", "relationship", "loss", "love", "home", "parent", "sister", "brother"],
    "Procedural/Tech": ["laboratory", "research", "technology", "machine", "experiment", "scientist", "scientific", "data", "computer", "discovery"]
}

def get_concept_profile(plot_text):
    text = str(plot_text).lower()
    profile = {}
    for bucket, keywords in CONCEPT_BUCKETS.items():
        count = sum(1 for word in keywords if word in text)
        profile[bucket] = count
    return profile

@app.post("/api/recommend", response_model=schemas.RecommendationResponse)
def suggest(req: schemas.RecommendationRequest):
    indices = df_slim.index[df_slim['Title'].str.lower() == req.title.lower()].tolist()
    if not indices:
        raise HTTPException(status_code=404, detail="Movie not found in the dataset.")

    movie_idx = indices[0]
    exclude_set = set(indices)

    # Weights from request or defaults
    W = req.weights or schemas.RecommendationWeights()

    target_year   = int(df_slim.iloc[movie_idx]['Release Year'])
    target_dir    = str(df_slim.iloc[movie_idx]['Director']).strip().lower()
    target_orig   = str(df_slim.iloc[movie_idx]['Origin/Ethnicity']).strip().lower()
    target_genre  = str(df_slim.iloc[movie_idx]['Genre']).lower()
    target_plot   = str(df_slim.iloc[movie_idx]['Plot'])

    # --- 1. Signal Calculation ---
    t_query = tfidf_plot.transform([target_plot[:300]])
    sim_plot = cosine_similarity(t_query, mat_plot).flatten()
    sim_genre = cosine_similarity(mat_genre[movie_idx], mat_genre).flatten()

    query_profile = get_concept_profile(target_plot)
    primary_query_bucket = max(query_profile, key=query_profile.get) if any(query_profile.values()) else None
    
    sim_concept = np.zeros(len(df_slim))
    for bucket, q_intensity in query_profile.items():
        if q_intensity == 0: continue
        keywords = CONCEPT_BUCKETS[bucket]
        mask = df_slim['Plot'].str.lower().str.contains('|'.join(keywords), na=False).values
        sim_concept[mask] += q_intensity
    
    if sim_concept.max() > 0:
        sim_concept = sim_concept / sim_concept.max()

    directors = df_slim['Director'].str.strip().str.lower()
    sim_dir   = np.where(directors == target_dir, 1.0, 0.7)
    
    origins   = df_slim['Origin/Ethnicity'].str.strip().str.lower()
    sim_orig  = np.where(origins == target_orig, 1.0, 0.7)

    years_array = df_slim['Release Year'].values
    sim_year    = np.maximum(0.0, 1.0 - (np.abs(years_array - target_year) / 50.0))

    # --- 2. Normalized Linear Fusion ---
    final_scores = (
        W.plot * sim_plot +
        W.genre * sim_genre +
        W.director * sim_dir +
        W.origin * sim_orig +
        W.year * sim_year +
        W.concept * sim_concept
    )

    # --- 3. Self-Exclusion & Ranking ---
    all_indices = np.arange(len(final_scores))
    candidate_indices = all_indices[~np.isin(all_indices, list(exclude_set))]
    candidate_scores  = final_scores[candidate_indices]

    top_local = np.argsort(-candidate_scores)[:req.top_n]
    top_indices = candidate_indices[top_local]
    top_scores  = candidate_scores[top_local]

    max_val = final_scores.max() if final_scores.max() > 0 else 1.0
    ui_scores = (top_scores / max_val)

    recommendations = []
    for i, s in zip(top_indices, ui_scores):
        contribs = {
            "Thematic Similarity": W.plot * sim_plot[i],
            "Genre Alignment": W.genre * sim_genre[i],
            f"Directed by {df_slim.iloc[i]['Director']}": W.director * sim_dir[i],
            "Regional Origin": W.origin * sim_orig[i],
            f"✨ {primary_query_bucket} Theme": W.concept * sim_concept[i]
        }
        explanation = max(contribs, key=contribs.get)

        recommendations.append(schemas.MovieRecommendation(
            title=str(df_slim.iloc[i]['Title']),
            release_year=int(df_slim.iloc[i]['Release Year']),
            genre=str(df_slim.iloc[i]['Genre']),
            origin=str(df_slim.iloc[i]['Origin/Ethnicity']),
            plot=str(df_slim.iloc[i]['Plot'])[:200] + "...",
            similarity_score=round(float(s), 4),
            explanation=explanation
        ))

    return schemas.RecommendationResponse(
        query_title=str(df_slim.iloc[movie_idx]['Title']),
        recommendations=recommendations
    )

@app.post("/api/search", response_model=schemas.SearchResponse)
def search(req: schemas.SearchRequest):
    query = req.query.lower().strip()
    titles = df_slim['Title'].unique()
    exact = [t for t in titles if t.lower() == query]
    prefix = [t for t in titles if t.lower().startswith(query) and t.lower() != query]
    contains = [t for t in titles if query in t.lower() and not t.lower().startswith(query)]
    combined = exact + sorted(prefix) + sorted(contains)
    return schemas.SearchResponse(results=combined[:req.limit])

@app.get("/api/movie/{title}")
def get_movie_details(title: str):
    matches = df_slim[df_slim['Title'].str.lower() == title.lower()]
    if matches.empty:
        raise HTTPException(status_code=404, detail="Movie not found")
    return matches.iloc[0].to_dict()

@app.get("/details")
def details_page():
    return FileResponse("static/details.html")

@app.get("/")
def root():
    return FileResponse("static/index.html")

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import joblib
import time
import os
import re

def load_data(path="wiki_movie_plots_deduped.csv"):
    print("Loading data...")
    df = pd.read_csv(path)
    for col in ['Genre', 'Director', 'Cast', 'Plot', 'Origin/Ethnicity']:
        df[col] = df[col].fillna('')

    return df

def genre_tokenizer(text):
    # Split by comma only — preserves space-separated compound genres like "science fiction"
    return [x.strip() for x in str(text).split(',')]

def build_multidim_models(df):
    start_time = time.time()

    print("Processing Genre dimension (comma-tokenized)...")
    tfidf_genre = TfidfVectorizer(tokenizer=genre_tokenizer, token_pattern=None)
    matrix_genre = normalize(tfidf_genre.fit_transform(df['Genre'].str.lower()))

    print("Processing Origin dimension...")
    tfidf_origin = TfidfVectorizer(stop_words='english')
    matrix_origin = normalize(tfidf_origin.fit_transform(df['Origin/Ethnicity'].str.lower()))

    print("Processing Cast dimension...")
    tfidf_cast = TfidfVectorizer(stop_words='english')
    matrix_cast = normalize(tfidf_cast.fit_transform(df['Cast'].str.lower()))

    # --- Reverting to Bigram TF-IDF for Plot ---
    print("Processing Plot dimension (Bigrams TF-IDF)...")
    tfidf_plot = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words='english',
        min_df=5,
        max_df=0.8,
        max_features=20000
    )
    matrix_plot = normalize(tfidf_plot.fit_transform(df['Plot'].str.lower()))

    print(f"Systems built in {(time.time() - start_time):.2f} seconds.")
    return (tfidf_genre, matrix_genre), (tfidf_origin, matrix_origin), (tfidf_cast, matrix_cast), (tfidf_plot, matrix_plot)

def main():
    os.makedirs("models", exist_ok=True)
    df = load_data()

    (m_genre, mat_g), (m_orig, mat_o), (m_cast, mat_c), (m_plot, mat_p) = build_multidim_models(df)

    print("Serializing models and matrices to disk...")

    # TF-IDF dimensions
    joblib.dump(m_genre, "models/tfidf_genre.pkl")
    joblib.dump(m_orig,  "models/tfidf_origin.pkl")
    joblib.dump(m_cast,  "models/tfidf_cast.pkl")
    joblib.dump(m_plot,  "models/tfidf_plot.pkl")

    joblib.dump(mat_g, "models/matrix_genre.pkl")
    joblib.dump(mat_o, "models/matrix_origin.pkl")
    joblib.dump(mat_c, "models/matrix_cast.pkl")
    joblib.dump(mat_p, "models/matrix_plot.pkl")

    # Store metadata including Raw Director and Plot
    df_slim = df[['Title', 'Release Year', 'Genre', 'Director', 'Origin/Ethnicity', 'Plot']]
    df_slim.to_pickle("models/dataset_metadata.pkl")

    print("Multi-Dimensional TF-IDF Pipeline Finished Successfully!")

if __name__ == "__main__":
    main()

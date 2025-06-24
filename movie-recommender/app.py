# Movie Recommendation System using MovieLens 100K dataset with Streamlit UI

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

# Load data
@st.cache_data
def load_data():
    ratings = pd.read_csv("ml-100k/u.data", sep='\t', names=["user_id", "item_id", "rating", "timestamp"])
    movies = pd.read_csv("ml-100k/u.item", sep='|', encoding='latin-1', usecols=list(range(24)),
                          names=["movie_id", "title", "release_date", "video_release_date", "IMDb_URL",
                                 "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
                                 "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
                                 "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"])
    df = pd.merge(ratings, movies, left_on='item_id', right_on='movie_id')
    return df

df = load_data()

# Content-Based Filtering (Using genres)
genre_cols = ["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama",
              "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]

genres_matrix = df.groupby('title')[genre_cols].mean()

def recommend_by_genre(title, top_n=5):
    if title not in genres_matrix.index:
        return []
    sims = cosine_similarity([genres_matrix.loc[title]], genres_matrix)[0]
    indices = np.argsort(sims)[::-1][1:top_n+1]
    return genres_matrix.iloc[indices].index.tolist()

# Collaborative Filtering
user_movie_matrix = df.pivot_table(index='user_id', columns='title', values='rating')
user_movie_matrix.fillna(0, inplace=True)
similarity_matrix = cosine_similarity(user_movie_matrix)

def recommend_movies_for_user(user_id, top_n=5):
    if user_id not in user_movie_matrix.index:
        return []
    user_ratings = user_movie_matrix.loc[user_id]
    similar_users = list(np.argsort(similarity_matrix[user_id-1])[::-1][1:6])
    similar_users_df = user_movie_matrix.iloc[similar_users]
    mean_ratings = similar_users_df.mean().sort_values(ascending=False)
    already_seen = set(user_ratings[user_ratings > 0].index)
    recommendations = [movie for movie in mean_ratings.index if movie not in already_seen][:top_n]
    return recommendations

# Streamlit UI
st.title("🎬 Movie Recommendation System")

option = st.radio("Choose Recommendation Type:", ("Content-Based", "Collaborative Filtering"))

if option == "Content-Based":
    selected_movie = st.selectbox("Select a movie to get similar recommendations:", genres_matrix.index.tolist())
    if st.button("Recommend by Genre"):
        results = recommend_by_genre(selected_movie)
        st.subheader("Top 5 Recommendations:")
        for idx, movie in enumerate(results, 1):
            st.write(f"{idx}. {movie}")

elif option == "Collaborative Filtering":
    selected_user = st.slider("Select a User ID:", min_value=1, max_value=943, value=5)
    if st.button("Recommend for User"):
        results = recommend_movies_for_user(selected_user)
        st.subheader(f"Top 5 Movie Recommendations for User {selected_user}:")
        for idx, movie in enumerate(results, 1):
            st.write(f"{idx}. {movie}")

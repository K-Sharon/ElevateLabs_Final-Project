```markdown
# 🎬 Movie Recommendation System (ML + Streamlit)

This project is a movie recommendation system using the **MovieLens 100K dataset**, combining:
- 🎯 Content-based filtering (based on movie genres)
- 👥 Collaborative filtering (based on user similarity)

---

## 📦 Features

- Input a movie → Get top 5 similar movies (genre-based)
- Input a user ID → Get top 5 unseen recommendations (user-based)
- Fully interactive UI via **Streamlit**
- Recommender logic powered by **cosine similarity** and **user-based filtering**

---

## 🛠 Tools Used

- Python
- Pandas & NumPy
- Scikit-learn
- Streamlit
- MovieLens 100K Dataset

---

## 🧠 Recommendation Logic

### 1️⃣ Content-Based Filtering

- Each movie is represented as a vector of its genres (e.g., Action, Comedy, Drama).
- Cosine similarity is used to compare the genre vector of the selected movie with all other movies.
- The top 5 most similar movies (excluding the input one) are recommended.

```python
cosine_similarity([genre_vector], genre_matrix)
```

### 2️⃣ Collaborative Filtering

- A **user-item matrix** is built where each row is a user and each column is a movie, filled with user ratings.
- Cosine similarity is calculated between users to find similar user preferences.
- The top 5 movies highly rated by similar users (and not yet watched by the target user) are recommended.

```python
cosine_similarity(user_rating_vector, user_rating_matrix)
```

---

## 🚀 How to Run the Project

```bash
pip install -r requirements.txt
streamlit run app.py
```

> Make sure the `ml-100k` dataset folder is in the same directory and contains `u.data` and `u.item`.


## 📜 Credits

- [MovieLens 100K Dataset](https://grouplens.org/datasets/movielens/100k/)
```

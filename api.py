import torch
import torch.nn as nn
from fastapi import FastAPI
import pandas as pd

from db import save_recommendation

# Load datasets
ratings = pd.read_csv("ml-latest-small/ratings.csv")
movies = pd.read_csv("ml-latest-small/movies.csv")

# -----------------------------
# Neural Recommendation Model
# -----------------------------
class RecommenderModel(nn.Module):

    def __init__(self, num_users, num_movies, embedding_dim=50):
        super().__init__()

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)

        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, user, movie):

        user_vec = self.user_embedding(user)
        movie_vec = self.movie_embedding(movie)

        x = torch.cat([user_vec, movie_vec], dim=1)

        return self.fc(x).squeeze()


# -----------------------------
# Load trained model
# -----------------------------
checkpoint = torch.load("recommender_model.pth")

model = RecommenderModel(
    checkpoint["num_users"],
    checkpoint["num_movies"]
)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI()


@app.get("/")
def home():

    return {
        "message": "Recommendation API is running"
    }


@app.get("/recommend/{user_id}")
def recommend(user_id: int):

    user_tensor = torch.tensor([user_id])

    movie_ids = ratings["movieId"].unique()

    scores = []

    # Predict ratings for movies
    for movie_id in movie_ids[:50]:

        movie_tensor = torch.tensor(
            [movie_id % checkpoint["num_movies"]]
        )

        with torch.no_grad():

            score = model(user_tensor, movie_tensor).item()

        scores.append((movie_id, score))

        # Save result to database
        save_recommendation(user_id, movie_id, score)

    # Sort by predicted score
    scores.sort(key=lambda x: x[1], reverse=True)

    top_movies = [m[0] for m in scores[:5]]

    recommendations = movies[
        movies["movieId"].isin(top_movies)
    ][["movieId", "title"]]

    return recommendations.to_dict(orient="records")
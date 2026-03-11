import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import mlflow
import mlflow.pytorch

# --------------------------------
# Load dataset
# --------------------------------
ratings = pd.read_csv("ml-latest-small/ratings.csv")

# Map user and movie IDs
user_ids = ratings["userId"].unique()
movie_ids = ratings["movieId"].unique()

user_map = {u: i for i, u in enumerate(user_ids)}
movie_map = {m: i for i, m in enumerate(movie_ids)}

ratings["user_idx"] = ratings["userId"].map(user_map)
ratings["movie_idx"] = ratings["movieId"].map(movie_map)

# --------------------------------
# Dataset class
# --------------------------------
class RatingDataset(Dataset):

    def __init__(self, df):

        self.users = torch.tensor(df["user_idx"].values)
        self.movies = torch.tensor(df["movie_idx"].values)
        self.ratings = torch.tensor(
            df["rating"].values,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):

        return (
            self.users[idx],
            self.movies[idx],
            self.ratings[idx]
        )


dataset = RatingDataset(ratings)

loader = DataLoader(
    dataset,
    batch_size=256,
    shuffle=True
)

# --------------------------------
# Neural Recommendation Model
# --------------------------------
class RecommenderModel(nn.Module):

    def __init__(self, num_users, num_movies, embedding_dim=50):

        super().__init__()

        self.user_embedding = nn.Embedding(
            num_users,
            embedding_dim
        )

        self.movie_embedding = nn.Embedding(
            num_movies,
            embedding_dim
        )

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


# --------------------------------
# Model setup
# --------------------------------
num_users = len(user_ids)
num_movies = len(movie_ids)

model = RecommenderModel(num_users, num_movies)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 3

# --------------------------------
# MLflow experiment
# --------------------------------
mlflow.set_experiment("movie_recommender")

with mlflow.start_run():

    mlflow.log_param("embedding_dim", 50)
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("epochs", epochs)

    # --------------------------------
    # Training loop
    # --------------------------------
    for epoch in range(epochs):

        total_loss = 0

        for user, movie, rating in loader:

            pred = model(user, movie)

            loss = criterion(pred, rating)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

        mlflow.log_metric("loss", total_loss, step=epoch)

    # --------------------------------
    # Save model locally
    # --------------------------------
    torch.save({
        "model_state_dict": model.state_dict(),
        "num_users": num_users,
        "num_movies": num_movies
    }, "recommender_model.pth")

    print("\nModel saved successfully")

    # --------------------------------
    # Save model in MLflow
    # --------------------------------
    mlflow.log_artifact("recommender_model.pth")

    # Save model locally
torch.save({
    "model_state_dict": model.state_dict(),
    "num_users": num_users,
    "num_movies": num_movies
}, "recommender_model.pth")

print("\nModel saved successfully")

# Log model artifact to MLflow
mlflow.log_artifact("recommender_model.pth")
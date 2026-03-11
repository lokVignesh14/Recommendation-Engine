import pandas as pd

# Load dataset
ratings = pd.read_csv("ml-latest-small/ratings.csv")
movies = pd.read_csv("ml-latest-small/movies.csv")

# Create user-item matrix
user_item_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
)

print("User-Item Matrix Shape:")
print(user_item_matrix.shape)

print("\nSample matrix:")
print(user_item_matrix.head())
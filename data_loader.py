import pandas as pd

# Load dataset
ratings = pd.read_csv("ml-latest-small/ratings.csv")
movies = pd.read_csv("ml-latest-small/movies.csv")

# Show dataset info
print("Ratings shape:", ratings.shape)
print("Movies shape:", movies.shape)

print("\nRatings sample:")
print(ratings.head())

print("\nMovies sample:")
print(movies.head())
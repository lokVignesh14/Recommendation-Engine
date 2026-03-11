import pandas as pd

# Load datasets
ratings = pd.read_csv("ml-latest-small/ratings.csv")
movies = pd.read_csv("ml-latest-small/movies.csv")

# Basic info
print("Ratings info:")
print(ratings.info())

print("\nMovies info:")
print(movies.info())

# Check missing values
print("\nMissing values in ratings:")
print(ratings.isnull().sum())

print("\nMissing values in movies:")
print(movies.isnull().sum())

# Unique users and movies
print("\nTotal users:", ratings["userId"].nunique())
print("Total movies rated:", ratings["movieId"].nunique())

# Average rating
print("\nAverage rating:", ratings["rating"].mean())

# Ratings distribution
print("\nRatings distribution:")
print(ratings["rating"].value_counts().sort_index())
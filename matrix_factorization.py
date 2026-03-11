import pandas as pd
from sklearn.decomposition import TruncatedSVD

# Load datasets
ratings = pd.read_csv("ml-latest-small/ratings.csv")
movies = pd.read_csv("ml-latest-small/movies.csv")

# Create user-item matrix
user_item_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
)

# Fill missing values
matrix_filled = user_item_matrix.fillna(0)

# Apply matrix factorization
svd = TruncatedSVD(n_components=20)

matrix_reduced = svd.fit_transform(matrix_filled)

# Reconstruct approximate ratings
predicted_ratings = matrix_reduced.dot(svd.components_)

predicted_df = pd.DataFrame(
    predicted_ratings,
    index=user_item_matrix.index,
    columns=user_item_matrix.columns
)


def recommend_movies(user_id, num_recommendations=5):

    user_predictions = predicted_df.loc[user_id]

    user_movies = user_item_matrix.loc[user_id]
    watched = user_movies[user_movies.notna()].index

    recommendations = user_predictions.drop(watched)

    top_movies = recommendations.sort_values(ascending=False).head(num_recommendations)

    return movies[movies["movieId"].isin(top_movies.index)][["movieId", "title"]]


print("\nMatrix Factorization Recommendations for User 1:\n")
print(recommend_movies(1))
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

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
user_item_matrix_filled = user_item_matrix.fillna(0)

# Compute user similarity
user_similarity = cosine_similarity(user_item_matrix_filled)

user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_item_matrix.index,
    columns=user_item_matrix.index
)


# Recommendation function
def recommend_movies(user_id, num_recommendations=5):

    # Find similar users
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)

    similar_users = similar_users.iloc[1:11]  # top 10 similar users

    # Movies watched by the target user
    user_movies = user_item_matrix.loc[user_id]
    user_watched = user_movies[user_movies.notna()].index

    # Movies watched by similar users
    similar_users_movies = user_item_matrix.loc[similar_users.index]

    # Average rating from similar users
    movie_scores = similar_users_movies.mean().sort_values(ascending=False)

    # Remove movies already watched
    recommendations = movie_scores.drop(user_watched)

    top_movies = recommendations.head(num_recommendations).index

    return movies[movies["movieId"].isin(top_movies)][["movieId", "title"]]


# Test recommendation
print("\nRecommended movies for User 1:\n")
print(recommend_movies(1))
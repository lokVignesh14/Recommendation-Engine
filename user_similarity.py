import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
ratings = pd.read_csv("ml-latest-small/ratings.csv")

# Create user-item matrix
user_item_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
)

# Replace missing values with 0
user_item_matrix_filled = user_item_matrix.fillna(0)

# Compute similarity between users
user_similarity = cosine_similarity(user_item_matrix_filled)

# Convert to DataFrame
user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_item_matrix.index,
    columns=user_item_matrix.index
)

print("User similarity matrix shape:")
print(user_similarity_df.shape)

print("\nSimilarity between first 5 users:")
print(user_similarity_df.iloc[:5, :5])

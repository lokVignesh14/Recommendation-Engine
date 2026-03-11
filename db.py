import os
import psycopg2

# Database connection details from environment variables
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "Vicky")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "Welcome2025@")


# Create connection
conn = psycopg2.connect(
    host=DB_HOST,
    port=DB_PORT,
    database=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD
)

cur = conn.cursor()


# Create table if it doesn't exist
cur.execute("""
CREATE TABLE IF NOT EXISTS recommendations (
    id SERIAL PRIMARY KEY,
    user_id INT,
    movie_id INT,
    score FLOAT
)
""")

conn.commit()


def save_recommendation(user_id, movie_id, score):
    """
    Save recommendation result into PostgreSQL
    """

    cur.execute(
        """
        INSERT INTO recommendations (user_id, movie_id, score)
        VALUES (%s, %s, %s)
        """,
        (
            int(user_id),      # fix numpy.int64 issue
            int(movie_id),     # convert to normal python int
            float(score)       # convert numpy float
        )
    )

    conn.commit()
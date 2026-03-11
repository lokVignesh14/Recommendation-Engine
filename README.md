🎬 Intelligent Movie Recommendation Engine














An AI-powered Movie Recommendation Engine that provides personalized movie suggestions using Collaborative Filtering and Neural Recommendation Models.

The system demonstrates a production-ready ML architecture combining:

Machine Learning models

REST API inference

Experiment tracking

Database storage

Containerized deployment

This architecture simulates how recommendation systems used by Netflix, Amazon, and Spotify operate.

🧩 Core Architecture

The recommendation system follows a structured ML pipeline.

User Request
│
▼
FastAPI Inference API
│
├── Collaborative Filtering Model
│    → User similarity recommendations
│
├── Neural Recommendation Model
│    → Deep learning rating prediction
│
└── Database Logging
     → PostgreSQL stores recommendations
🚀 Key AI Features
🎯 Collaborative Filtering

Finds similar users based on historical movie ratings.

Technique used:

User–User Similarity
Cosine Similarity

Example:

If User A and User B like similar movies, the system recommends movies liked by B to A.

🧠 Neural Recommendation Model

A deep learning model predicts user preferences.

Architecture:

User Embedding
      +
Movie Embedding
      ↓
Neural Network
      ↓
Predicted Rating Score

This allows the model to learn latent relationships between users and movies.

⚡ API-based Recommendation System

Users interact with the recommendation engine through a REST API.

Example endpoint:

GET /recommend/{user_id}

Example request:

http://localhost:8000/recommend/1

Example response:

{
  "user_id": 1,
  "recommended_movies": [
    "Toy Story",
    "The Matrix",
    "The Dark Knight"
  ]
}
📊 MLflow Experiment Tracking

MLflow is used to track machine learning experiments including:

Model parameters

Training metrics

Experiment runs

Model artifacts

Run MLflow UI:

mlflow ui

Open in browser:

http://localhost:5000
🏗️ System Components
AI Layer

Collaborative Filtering

Neural Recommendation Model

Similarity computation

Backend Layer

FastAPI

REST API endpoints

Recommendation inference

Data Layer

PostgreSQL database

MovieLens dataset

Recommendation logging

MLOps Layer

MLflow experiment tracking

Docker containerization

⚙️ Technology Stack
AI / ML

Python

NumPy

Pandas

Scikit-learn

PyTorch

Backend

FastAPI

Python Async APIs

Database

PostgreSQL

Psycopg2

MLOps

MLflow

Docker

📦 Installation
Clone Repository
git clone https://github.com/lokVignesh14/Recommendation-Engine.git
cd Recommendation-Engine
Create Virtual Environment
python -m venv venv

Windows

venv\Scripts\activate

Linux / Mac

source venv/bin/activate
Install Dependencies
pip install -r requirements.txt
⚙️ Configuration

Create a .env file in the root directory.

DB_HOST=localhost
DB_PORT=5432
DB_NAME=Vicky
DB_USER=postgres
DB_PASSWORD=yourpassword
▶️ Running the Server
uvicorn api:app --reload

Server runs at:

http://127.0.0.1:8000

API documentation:

http://127.0.0.1:8000/docs
🐳 Docker Deployment

Build the Docker image:

docker build -t movie-recommender .

Run the container:

docker run --network ai-network -p 8000:8000 \
-e DB_HOST=postgres-db \
-e DB_PORT=5432 \
-e DB_NAME=Vicky \
-e DB_USER=postgres \
-e DB_PASSWORD=yourpassword \
movie-recommender
📂 Project Structure
recommendation-engine
│
├── api.py
├── db.py
├── data_loader.py
├── neural_recommender.py
├── recommend_movies.py
├── user_similarity.py
│
├── requirements.txt
├── Dockerfile
├── README.md
🎯 Use Cases

Personalized movie recommendation systems

AI-powered streaming platforms

Content discovery engines

Machine learning portfolio projects

🔮 Future Improvements

Matrix Factorization models

Transformer-based recommendation systems

Online learning recommendation pipelines

Cloud deployment (AWS / GCP)

👨‍💻 Author

Lok Vignesh

AI / Machine Learning Engineer

LinkedIn
https://linkedin.com/in/lok-vignesh-b-3a3454227

Email
lokvignesh1b@gmail.com

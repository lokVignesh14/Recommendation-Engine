# 🎬 AI Personalized Recommendation Engine

An end-to-end **AI-powered Movie Recommendation System** built using Machine Learning and Deep Learning techniques.

This project demonstrates how modern recommendation systems work by combining:

• Collaborative Filtering
• Neural Recommendation Models
• API-based inference
• Experiment tracking
• Containerized deployment

The system recommends movies to users based on historical ratings and stores predictions in a database.

---

# 🚀 Project Overview

Recommendation systems power platforms like:

• Netflix
• Amazon
• Spotify
• YouTube

This project replicates a simplified version of those systems using the **MovieLens dataset**.

The pipeline includes:

```
Dataset
   ↓
Data Processing
   ↓
User Similarity Model
   ↓
Neural Recommendation Model
   ↓
FastAPI Inference API
   ↓
PostgreSQL Storage
   ↓
Docker Deployment
```

---

# 📊 Models Used

### 1️⃣ Collaborative Filtering

Finds similarity between users based on movie ratings.

Technique used:

```
User–User Similarity
Cosine Similarity
```

Used for generating baseline recommendations.

---

### 2️⃣ Neural Recommendation Model

Deep learning model built with **PyTorch**.

Architecture:

```
User Embedding
      +
Movie Embedding
      ↓
Neural Network
      ↓
Predicted Rating Score
```

This model learns latent relationships between users and movies.

---

# 🛠 Tech Stack

### Programming

Python

### Machine Learning

NumPy
Pandas
Scikit-learn
PyTorch

### Backend

FastAPI

### Database

PostgreSQL

### MLOps

MLflow

### Deployment

Docker

---

# 📂 Project Structure

```
recommendation-engine
│
├── api.py
├── db.py
├── data_loader.py
├── user_similarity.py
├── recommend_movies.py
├── neural_recommender.py
│
├── requirements.txt
├── Dockerfile
├── README.md
```

---

# ⚙️ Features

• Movie recommendation based on user behavior
• Neural network-based prediction model
• Experiment tracking using MLflow
• FastAPI inference endpoint
• PostgreSQL database storage
• Docker containerized deployment

---

# 🔌 API Endpoint

### Get Recommendations

```
GET /recommend/{user_id}
```

Example:

```
http://localhost:8000/recommend/1
```

Response:

```
{
 "user_id": 1,
 "recommended_movies": [
   "Toy Story",
   "The Matrix",
   "The Dark Knight"
 ]
}
```

---

# 📊 MLflow Experiment Tracking

MLflow is used to track:

```
Model parameters
Training loss
Model artifacts
Experiment runs
```

Run MLflow UI:

```
mlflow ui
```

Then open:

```
http://localhost:5000
```

---

# 🐳 Docker Deployment

Build the Docker image:

```
docker build -t movie-recommender .
```

Run the container:

```
docker run --network ai-network -p 8000:8000 \
-e DB_HOST=postgres-db \
-e DB_PORT=5432 \
-e DB_NAME=Vicky \
-e DB_USER=postgres \
-e DB_PASSWORD=yourpassword \
movie-recommender
```

---

# 🗄 Database

PostgreSQL is used to store recommendation results.

Table structure:

```
recommendations

id
user_id
movie_id
score
```

---

# 📈 Future Improvements

• Matrix Factorization models
• Transformer-based recommendation models
• Online learning pipelines
• Feature store integration
• Cloud deployment (AWS / GCP)

---

# 🎯 Learning Outcomes

Through this project:

• Built an end-to-end ML system
• Implemented neural recommendation models
• Deployed ML APIs with FastAPI
• Tracked experiments using MLflow
• Containerized ML services with Docker

---

# 👨‍💻 Author

Lok Vignesh

AI / Machine Learning Engineer

LinkedIn
https://linkedin.com/in/lok-vignesh-b-3a3454227

Email
[lokvignesh1b@gmail.com](mailto:lokvignesh1b@gmail.com)

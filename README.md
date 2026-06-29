# Customer Churn Prediction

An end-to-end machine learning project developed during my Data Science internship to identify customers at high risk of churning. This tool provides predictive analysis along with actionable retention recommendations.

## Features
* **Machine Learning Model:** Utilizes a Random Forest Classifier trained on customer demographic, service, and account data.
* **Backend API:** A Flask-based REST API serving predictions and sample data.
* **Interactive Frontend:** A responsive web interface allowing users to input customer details and instantly view churn risk, probability, and tailored recommendations.

## Tech Stack
* **Data Science & Machine Learning:** Python, Scikit-learn, Pandas, NumPy
* **Backend:** Flask, Flask-CORS, Joblib
* **Frontend:** HTML5, CSS3, Vanilla JavaScript

## How to Run Locally

### 1. Backend Setup
Navigate to the backend directory and install the required packages:
`pip install -r requirements.txt`

Run the Flask server:
`python app.py`
The server will start on `http://localhost:5000`.

### 2. Frontend Setup
Simply open the `index.html` file in any modern web browser. The frontend will automatically connect to the local Flask server to generate predictions.

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from model import predictor
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/')
def home():
    return jsonify({
        "message": "Customer Churn Prediction API",
        "status": "running",
        "endpoints": {
            "/predict": "POST - Predict customer churn",
            "/health": "GET - API health check",
            "/sample": "GET - Get sample customer data"
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "model_loaded": predictor.model is not None})

@app.route('/sample', methods=['GET'])
def get_sample_data():
    """Get sample customer data for testing"""
    sample_customer = {
        "age": 45,
        "gender": "Female",
        "tenure": 24,
        "monthly_charges": 65.50,
        "total_charges": 1572.00,
        "contract_type": "One year",
        "internet_service": "Fiber optic",
        "online_security": "Yes",
        "tech_support": "Yes",
        "streaming_tv": "Yes",
        "monthly_usage_gb": 350,
        "support_calls": 2,
        "payment_delay": 0
    }
    return jsonify(sample_customer)

@app.route('/predict', methods=['POST'])
def predict_churn():
    """Predict customer churn based on input data"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Validate required fields
        required_fields = ['age', 'tenure', 'monthly_charges', 'contract_type']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Make prediction
        will_churn, probability = predictor.predict_churn(data)
        
        # Prepare response
        response = {
            "prediction": "High Risk" if will_churn else "Low Risk",
            "churn_probability": round(probability * 100, 2),
            "will_churn": will_churn,
            "confidence": "high" if probability > 0.7 or probability < 0.3 else "medium",
            "recommendation": get_recommendation(will_churn, probability, data)
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def get_recommendation(will_churn, probability, customer_data):
    """Generate recommendations based on prediction"""
    if will_churn:
        recommendations = ["Consider offering loyalty discount", "Reach out with personalized offer"]
        
        if customer_data.get('contract_type') == 'Month-to-month':
            recommendations.append("Offer contract upgrade with better rates")
        
        if customer_data.get('support_calls', 0) > 3:
            recommendations.append("Provide dedicated customer support")
        
        if customer_data.get('monthly_charges', 0) > 80:
            recommendations.append("Consider price adjustment or bundle offer")
    else:
        recommendations = ["Customer shows good engagement", "Maintain current service quality"]
    
    return recommendations

if __name__ == '__main__':
    # Load pre-trained model
    if not predictor.load_model():
        print("Training new model...")
        predictor.train_model()
        predictor.save_model()
    
    print("Model loaded successfully!")
    app.run(debug=True, host='0.0.0.0', port=5000)
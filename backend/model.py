import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

class ChurnPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def generate_sample_data(self, n_samples=1000):
        """Generate synthetic customer data for demonstration"""
        np.random.seed(42)
        
        data = {
            'customer_id': range(1, n_samples + 1),
            'age': np.random.randint(18, 80, n_samples),
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'tenure': np.random.randint(1, 72, n_samples),  # months
            'monthly_charges': np.random.uniform(20, 100, n_samples).round(2),
            'total_charges': np.random.uniform(50, 5000, n_samples).round(2),
            'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
            'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
            'online_security': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'tech_support': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'streaming_tv': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'monthly_usage_gb': np.random.randint(50, 1000, n_samples),
            'support_calls': np.random.randint(0, 10, n_samples),
            'payment_delay': np.random.randint(0, 3, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Create churn based on features
        churn_prob = (
            (df['tenure'] < 12) * 0.3 +
            (df['monthly_charges'] > 80) * 0.2 +
            (df['contract_type'] == 'Month-to-month') * 0.3 +
            (df['support_calls'] > 5) * 0.4 +
            (df['payment_delay'] > 1) * 0.3 +
            (df['tech_support'] == 'No') * 0.2
        )
        
        df['churn'] = (churn_prob > 0.5).astype(int)
        
        return df
    
    def preprocess_data(self, df):
        """Preprocess the data for training"""
        # Copy dataframe
        df_processed = df.copy()
        
        # Encode categorical variables
        categorical_columns = ['gender', 'contract_type', 'internet_service', 
                              'online_security', 'tech_support', 'streaming_tv']
        
        for col in categorical_columns:
            self.label_encoders[col] = LabelEncoder()
            df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col])
        
        # Features for training
        feature_columns = ['age', 'tenure', 'monthly_charges', 'total_charges', 
                          'monthly_usage_gb', 'support_calls', 'payment_delay'] + categorical_columns
        
        X = df_processed[feature_columns]
        y = df_processed['churn']
        
        return X, y, feature_columns
    
    def train_model(self):
        """Train the Random Forest model"""
        # Generate sample data
        df = self.generate_sample_data(1000)
        
        # Preprocess data
        X, y, feature_columns = self.preprocess_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale numerical features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model trained with accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))
        
        return accuracy, feature_columns
    
    def predict_churn(self, customer_data):
        """Predict churn for a single customer"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Convert input to dataframe
        df_input = pd.DataFrame([customer_data])
        
        # Encode categorical variables
        categorical_columns = ['gender', 'contract_type', 'internet_service', 
                              'online_security', 'tech_support', 'streaming_tv']
        
        for col in categorical_columns:
            if col in df_input.columns:
                if col in self.label_encoders:
                    # Handle unseen labels
                    try:
                        df_input[col] = self.label_encoders[col].transform([df_input[col].iloc[0]])[0]
                    except ValueError:
                        # If label not seen during training, use most frequent
                        df_input[col] = 0
        
        # Define feature columns in correct order
        feature_columns = ['age', 'tenure', 'monthly_charges', 'total_charges', 
                          'monthly_usage_gb', 'support_calls', 'payment_delay'] + categorical_columns
        
        # Ensure all columns are present
        for col in feature_columns:
            if col not in df_input.columns:
                df_input[col] = 0
        
        # Reorder columns
        df_input = df_input[feature_columns]
        
        # Scale features
        X_scaled = self.scaler.transform(df_input)
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0][1]
        
        return bool(prediction), float(probability)
    
    def save_model(self, model_path='churn_model.joblib'):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders
        }
        joblib.dump(model_data, model_path)
    
    def load_model(self, model_path='churn_model.joblib'):
        """Load a trained model"""
        if os.path.exists(model_path):
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoders = model_data['label_encoders']
            return True
        return False

# Create and train model instance
predictor = ChurnPredictor()
accuracy, features = predictor.train_model()
predictor.save_model()
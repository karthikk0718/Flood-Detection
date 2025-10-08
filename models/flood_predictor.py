import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class FloodPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.model_type = 'random_forest'  # or 'logistic_regression'
        self.feature_names = [
            'rainfall', 'river_level', 'temperature', 'humidity', 'wind_speed',
            'previous_rainfall', 'soil_moisture', 'elevation'
        ]
        self.model_path = 'flood_model.pkl'
        self.scaler_path = 'scaler.pkl'
    
    def generate_synthetic_training_data(self, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic training data for demonstration purposes
        In production, this would be replaced with real historical data
        """
        np.random.seed(42)  # For reproducibility
        
        # Generate feature data
        rainfall = np.random.exponential(scale=15, size=n_samples)
        river_level = np.random.normal(loc=5, scale=2, size=n_samples)
        temperature = np.random.normal(loc=28, scale=5, size=n_samples)
        humidity = np.random.normal(loc=70, scale=15, size=n_samples)
        wind_speed = np.random.exponential(scale=10, size=n_samples)
        previous_rainfall = np.random.exponential(scale=10, size=n_samples)
        soil_moisture = np.random.normal(loc=50, scale=20, size=n_samples)
        elevation = np.random.uniform(low=0, high=1000, size=n_samples)
        
        # Ensure realistic ranges
        rainfall = np.clip(rainfall, 0, 200)
        river_level = np.clip(river_level, 0, 20)
        temperature = np.clip(temperature, 15, 45)
        humidity = np.clip(humidity, 30, 100)
        wind_speed = np.clip(wind_speed, 0, 50)
        previous_rainfall = np.clip(previous_rainfall, 0, 300)
        soil_moisture = np.clip(soil_moisture, 0, 100)
        
        # Create feature matrix
        X = np.column_stack([
            rainfall, river_level, temperature, humidity, wind_speed,
            previous_rainfall, soil_moisture, elevation
        ])
        
        # Generate flood labels based on realistic conditions
        flood_probability = (
            0.3 * (rainfall / 50) +  # Higher rainfall increases flood risk
            0.25 * (river_level / 10) +  # Higher river levels increase risk
            0.15 * (previous_rainfall / 100) +  # Previous rainfall contributes
            0.1 * ((100 - elevation) / 100) +  # Lower elevation increases risk
            0.1 * (soil_moisture / 100) +  # Higher soil moisture increases risk
            0.1 * (humidity / 100) -  # Higher humidity slightly increases risk
            0.05 * (wind_speed / 25)  # Higher wind speed slightly decreases risk
        )
        
        # Add some randomness
        flood_probability += np.random.normal(0, 0.1, size=n_samples)
        flood_probability = np.clip(flood_probability, 0, 1)
        
        # Convert to binary labels
        y = (flood_probability > 0.5).astype(int)
        
        return X, y
    
    def load_or_train_model(self):
        """Load existing model or train new one if not found"""
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            try:
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                self.is_trained = True
                print("Model loaded successfully")
                return
            except Exception as e:
                print(f"Error loading model: {e}")
        
        # Train new model
        print("Training new model...")
        X, y = self.generate_synthetic_training_data(n_samples=2000)
        self.train(X, y)
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the flood prediction model"""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Choose model based on model_type
            if self.model_type == 'random_forest':
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    class_weight='balanced'
                )
            else:
                self.model = LogisticRegression(
                    random_state=42,
                    class_weight='balanced',
                    max_iter=1000
                )
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            train_score = self.model.score(X_train_scaled, y_train)
            test_score = self.model.score(X_test_scaled, y_test)
            
            print(f"Training accuracy: {train_score:.3f}")
            print(f"Test accuracy: {test_score:.3f}")
            
            # Cross-validation
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
            print(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            
            # Save model
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            
            self.is_trained = True
            print("Model training completed and saved")
            
        except Exception as e:
            print(f"Error training model: {e}")
            raise e
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make flood predictions"""
        if not self.is_trained or self.model is None:
            raise ValueError("Model is not trained yet")
        
        try:
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled)
            return predictions
        except Exception as e:
            raise ValueError(f"Error making predictions: {e}")
    
    def predict_probability(self, X: np.ndarray) -> np.ndarray:
        """Get flood probabilities"""
        if not self.is_trained or self.model is None:
            raise ValueError("Model is not trained yet")
        
        try:
            X_scaled = self.scaler.transform(X)
            probabilities = self.model.predict_proba(X_scaled)
            return probabilities[:, 1] if probabilities.ndim > 1 else probabilities  # Return probability of flood (class 1)
        except Exception as e:
            raise ValueError(f"Error getting probabilities: {e}")
    
    def get_model_metrics(self) -> Dict:
        """Get comprehensive model performance metrics"""
        if not self.is_trained or self.model is None:
            return {}
        
        try:
            # Generate test data for evaluation
            X_test, y_test = self.generate_synthetic_training_data(n_samples=500)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Predictions
            y_pred = self.model.predict(X_test_scaled)
            y_pred_proba_all = self.model.predict_proba(X_test_scaled)
            y_pred_proba = y_pred_proba_all[:, 1] if y_pred_proba_all.ndim > 1 else y_pred_proba_all
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0.0),
                'recall': recall_score(y_test, y_pred, zero_division=0.0),
                'f1_score': f1_score(y_test, y_pred, zero_division=0.0),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            }
            
            # ROC curve data
            if len(set(y_test)) > 1:  # Only if we have both classes
                fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
                metrics['roc_data'] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'auc': roc_auc_score(y_test, y_pred_proba)
                }
            
            # Feature importance (for Random Forest)
            if hasattr(self.model, 'feature_importances_'):
                metrics['feature_importance'] = self.model.feature_importances_.tolist()
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return {}
    
    def get_feature_importance(self) -> Optional[List[Tuple[str, float]]]:
        """Get feature importance scores"""
        if not self.is_trained or self.model is None:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            importance_pairs = list(zip(self.feature_names, self.model.feature_importances_))
            return sorted(importance_pairs, key=lambda x: x[1], reverse=True)
        
        return None

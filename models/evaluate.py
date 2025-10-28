"""
ANN Savings Model Evaluation Script

This script evaluates the performance of the trained neural network model
using comprehensive metrics to assess prediction accuracy across all outputs:
1. Spending percentage predictions
2. Saving percentage predictions  
3. Goal achievement classification

The evaluation provides insights into model reliability and helps determine
if the model is ready for production deployment in the Flask web application.

Metrics Calculated:
- Mean Absolute Error (MAE) for regression outputs
- R² Score for explained variance assessment
- Classification accuracy for goal achievement

Author: ANN Savings Calculator Team
Created: 2025
"""

import json
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score


def load_test_data():
    """
    Load the complete dataset for comprehensive model evaluation.
    
    Returns:
    --------
    tuple: (X, y) - Features and targets for evaluation
    """
    print("📂 Loading evaluation dataset...")
    
    # Load the original dataset used for training
    df = pd.read_csv("data/raw_data.csv")
    
    # Extract features (inputs to the model)
    X = df[["Income", "Spending", "Saving_Goal"]].copy()
    
    # Extract targets (expected outputs from the model)
    y = df[["Spending_Percent", "Saving_Percent", "Goal_Met"]].copy()
    
    print(f"📊 Evaluation samples: {len(df)}")
    return X, y


def load_model_artifacts():
    """
    Load the trained model and preprocessing scaler.
    
    Returns:
    --------
    tuple: (model, scaler) - Trained neural network and feature scaler
    """
    print("🧠 Loading trained model and scaler...")
    
    # Load the feature scaler used during training
    scaler = joblib.load("models/scaler.pkl")
    
    # Load the trained Keras model
    model = tf.keras.models.load_model("models/savings_model.keras")
    
    print("✅ Model artifacts loaded successfully")
    return model, scaler


def generate_predictions(model, scaler, X):
    """
    Generate predictions using the trained model.
    
    Parameters:
    -----------
    model : tf.keras.Model
        Trained neural network model
    scaler : sklearn.preprocessing.MinMaxScaler
        Fitted feature scaler
    X : pandas.DataFrame
        Input features for prediction
        
    Returns:
    --------
    tuple: (spend_hat, save_hat, goal_hat) - Model predictions
    """
    print("🔮 Generating model predictions...")
    
    # Scale input features using the same scaler from training
    X_scaled = scaler.transform(X)
    
    # Generate predictions for all samples
    predictions = model.predict(X_scaled, verbose=0)
    
    # Extract individual prediction components
    spend_hat = predictions[:, 0]  # Spending percentage predictions
    save_hat = predictions[:, 1]   # Saving percentage predictions
    
    # Convert goal achievement probabilities to binary classification
    # Threshold at 0.5: >= 0.5 means goal achieved (1), < 0.5 means not achieved (0)
    goal_hat = (predictions[:, 2] >= 0.5).astype(int)
    
    print(f"✅ Generated {len(predictions)} predictions")
    return spend_hat, save_hat, goal_hat


def calculate_metrics(y_true, spend_hat, save_hat, goal_hat):
    """
    Calculate comprehensive evaluation metrics for model performance.
    
    Parameters:
    -----------
    y_true : pandas.DataFrame
        True target values
    spend_hat : numpy.ndarray
        Predicted spending percentages
    save_hat : numpy.ndarray
        Predicted saving percentages  
    goal_hat : numpy.ndarray
        Predicted goal achievements (binary)
        
    Returns:
    --------
    dict: Comprehensive metrics dictionary
    """
    print("📊 Calculating evaluation metrics...")
    
    # Mean Absolute Error for spending predictions
    mae_spend = mean_absolute_error(y_true["Spending_Percent"], spend_hat)
    
    # Mean Absolute Error for saving predictions
    mae_save = mean_absolute_error(y_true["Saving_Percent"], save_hat)
    
    # R² Score (coefficient of determination) for spending predictions
    # Measures how well the model explains variance in spending patterns
    r2_spend = r2_score(y_true["Spending_Percent"], spend_hat)
    
    # R² Score for saving predictions
    # Measures how well the model explains variance in saving patterns
    r2_save = r2_score(y_true["Saving_Percent"], save_hat)
    
    # Classification accuracy for goal achievement predictions
    acc_goal = accuracy_score(y_true["Goal_Met"], goal_hat)
    
    # Compile all metrics into a comprehensive report
    metrics = {
        "mae_spending_pct": float(mae_spend),
        "mae_saving_pct": float(mae_save),
        "r2_spending_pct": float(r2_spend),
        "r2_saving_pct": float(r2_save),
        "goal_met_accuracy": float(acc_goal),
    }
    
    return metrics


def save_metrics(metrics):
    """
    Save evaluation metrics to JSON file for future reference.
    
    Parameters:
    -----------
    metrics : dict
        Calculated evaluation metrics
    """
    print("💾 Saving evaluation metrics...")
    
    # Save metrics to JSON file for documentation and monitoring
    with open("models/extra_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print("✅ Metrics saved to models/extra_metrics.json")


def demonstrate_prediction(model, scaler):
    """
    Demonstrate model prediction with a sample case.
    
    Parameters:
    -----------
    model : tf.keras.Model
        Trained neural network model
    scaler : sklearn.preprocessing.MinMaxScaler
        Fitted feature scaler
    """
    print("\n🎯 Sample Prediction Demonstration:")
    
    # Create a sample financial scenario
    # Income: ₹50,000, Spending: ₹30,000, Savings Goal: 40%
    sample = np.array([[50000, 30000, 0.40]])
    
    print(f"📋 Sample Input:")
    print(f"   Income: ₹{sample[0,0]:,.0f}")
    print(f"   Spending: ₹{sample[0,1]:,.0f}")
    print(f"   Savings Goal: {sample[0,2]*100:.0f}%")
    
    # Generate prediction for the sample
    prediction = model.predict(scaler.transform(sample), verbose=0)[0]
    
    # Extract prediction components
    spending_pct = prediction[0] * 100
    saving_pct = prediction[1] * 100
    goal_achieved = "YES" if prediction[1] >= sample[0,2] else "NO"
    
    print(f"\n🔮 Model Predictions:")
    print(f"   Predicted Spending: {spending_pct:.1f}%")
    print(f"   Predicted Saving: {saving_pct:.1f}%")
    print(f"   Goal Achievement: {goal_achieved}")


def main():
    """
    Main evaluation pipeline execution.
    
    Pipeline Steps:
    1. Load test data and model artifacts
    2. Generate predictions on full dataset
    3. Calculate comprehensive metrics
    4. Save results and demonstrate prediction
    """
    print("=" * 60)
    print("📊 ANN SAVINGS MODEL EVALUATION")
    print("=" * 60)
    
    try:
        # Step 1: Load data and model artifacts
        X, y = load_test_data()
        model, scaler = load_model_artifacts()
        
        # Step 2: Generate predictions
        spend_hat, save_hat, goal_hat = generate_predictions(model, scaler, X)
        
        # Step 3: Calculate evaluation metrics
        metrics = calculate_metrics(y, spend_hat, save_hat, goal_hat)
        
        # Step 4: Display and save results
        print("\n📈 EVALUATION RESULTS:")
        print("=" * 40)
        print(json.dumps(metrics, indent=2))
        
        save_metrics(metrics)
        
        # Step 5: Demonstrate with sample prediction
        demonstrate_prediction(model, scaler)
        
        print("\n" + "=" * 60)
        print("✅ EVALUATION COMPLETED SUCCESSFULLY!")
        print(f"🎯 Goal Accuracy: {metrics['goal_met_accuracy']:.3f}")
        print(f"📊 Spending MAE: {metrics['mae_spending_pct']:.4f}")
        print(f"💰 Saving MAE: {metrics['mae_saving_pct']:.4f}")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
        raise


# Script execution entry point
if __name__ == '__main__':
    """
    Execute the evaluation pipeline when script is run directly.
    
    Usage:
        python evaluate.py
        
    Prerequisites:
        - Trained model must exist at 'models/savings_model.keras'
        - Scaler must exist at 'models/scaler.pkl'
        - Test data must exist at 'data/raw_data.csv'
    """
    main()

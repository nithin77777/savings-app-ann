"""
ANN Savings Calculator - Flask Web Application

This Flask application serves as the web interface for the ANN-based savings
prediction system. It provides a user-friendly web interface where users can
input their financial information and receive AI-powered insights about their
spending patterns, saving potential, and goal achievement likelihood.

Key Features:
- RESTful API for financial predictions
- Interactive web interface with real-time charts
- Input validation and error handling
- Personalized recommendations and rewards system
- Responsive design for all devices

Endpoints:
- GET /: Main application interface (HTML)
- POST /api/predict: Financial prediction API (JSON)

Author: ANN Savings Calculator Team
Created: 2025
"""

from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import joblib
import json
import os
import tensorflow as tf

# Initialize Flask application
app = Flask(__name__)

# Configuration: Model and scaler file paths
# Environment variables allow for flexible deployment configurations
MODEL_PATH = os.environ.get("MODEL_PATH", "models/savings_model.keras")
SCALER_PATH = os.environ.get("SCALER_PATH", "models/scaler.pkl")

# Load pre-trained model and scaler at application startup
# This ensures fast response times for API requests
print("🚀 Loading ANN model and scaler...")
model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print("✅ Model artifacts loaded successfully")

def predict_payload(payload):
    """
    Process user input and generate AI-powered financial predictions.
    
    This function serves as the core prediction engine that:
    1. Validates and processes user input
    2. Applies neural network model for predictions
    3. Generates personalized recommendations
    4. Calculates rewards for goal achievement
    
    Parameters:
    -----------
    payload : dict
        User input containing:
        - income: Monthly income in INR
        - spending: Monthly spending in INR  
        - savings_goal: Target savings percentage (1-99)
    
    Returns:
    --------
    dict: Comprehensive prediction results with recommendations
    
    Raises:
    -------
    ValueError: For invalid input values or ranges
    """
    
    # === INPUT VALIDATION AND PREPROCESSING ===
    try:
        # Extract and convert input values to float
        income = float(payload.get("income"))
        spending = float(payload.get("spending"))
        goal_pct = float(payload.get("savings_goal"))  # e.g. 40 means 40%
    except (TypeError, ValueError):
        raise ValueError("income, spending, savings_goal must be numeric")

    # Validate income constraints
    if income <= 0:
        raise ValueError("income must be > 0")
    
    # Validate spending constraints  
    if spending < 0:
        raise ValueError("spending must be >= 0")
    
    # Validate savings goal range (1-99%)
    if goal_pct <= 0 or goal_pct >= 100:
        raise ValueError("savings_goal must be in (0, 100)")

    # === NEURAL NETWORK PREDICTION ===
    # Prepare input array for model prediction
    # Format: [income, spending, goal_as_decimal]
    X = np.array([[income, spending, goal_pct/100.0]], dtype=float)
    
    # Apply feature scaling (same scaling used during training)
    X_scaled = scaler.transform(X)
    
    # Generate predictions using trained ANN model
    predictions = model.predict(X_scaled, verbose=0)[0]
    
    # Extract prediction components
    spend_p = float(predictions[0])     # Predicted spending percentage
    save_p = float(predictions[1])      # Predicted saving percentage  
    goal_score = float(predictions[2])  # Goal achievement probability
    
    # Determine if savings goal is achieved
    achieved = save_p >= (goal_pct/100.0)

    # === PERSONALIZED RECOMMENDATIONS ===
    if achieved:
        # Positive reinforcement for goal achievement
        recommendation = f"You're on track for your {int(goal_pct)}% savings goal."
    else:
        # Calculate required spending reduction to meet goal
        delta = round((goal_pct/100.0 - save_p)*100, 2)
        recommendation = f"Reduce spending by {delta}% to achieve your {int(goal_pct)}% savings goal"

    # === REWARD SYSTEM ===
    # Reward calculation: Goal percentage * 0.1% * Income
    # This incentivizes higher savings goals and rewards achievement
    if achieved:
        reward = round((goal_pct/100.0) * 0.001 * income, 2)
        note = (f'Congrats! You have achieved your goal - Your Reward is "{reward} INR", '
                f'you are free to spend this wherever you want. Cheers!')
    else:
        note = "You didn't achieve your goal. All the best for next month!"

    # === STRUCTURED RESPONSE ===
    return {
        "spending_percentage": round(spend_p * 100, 2),
        "savings_percentage": round(save_p * 100, 2), 
        "goal_achieved": achieved,
        "recommendation": recommendation,
        "note": note
    }

@app.route("/", methods=["GET"])
def index():
    """
    Serve the main application interface.
    
    This endpoint renders the HTML template containing the interactive
    form, result display areas, and chart visualization components.
    
    Returns:
    --------
    str: Rendered HTML template for the web interface
    """
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    API endpoint for financial predictions.
    
    This RESTful endpoint accepts JSON payloads with financial data
    and returns AI-generated predictions and recommendations.
    
    Expected JSON payload:
    {
        "income": 50000,
        "spending": 30000, 
        "savings_goal": 40
    }
    
    Returns:
    --------
    JSON response:
    - Success: {"ok": true, "result": {...}}
    - Error: {"ok": false, "error": "error message"}
    
    HTTP Status Codes:
    - 200: Successful prediction
    - 400: Invalid input or validation error
    """
    try:
        # Parse JSON payload from request
        data = request.get_json(force=True)
        
        # Generate predictions using the ANN model
        result = predict_payload(data)
        
        # Return successful response with predictions
        return jsonify({"ok": True, "result": result})
        
    except Exception as e:
        # Handle validation errors and model failures
        return jsonify({"ok": False, "error": str(e)}), 400


# === APPLICATION ENTRY POINT ===
if __name__ == "__main__":
    """
    Development server configuration.
    
    Starts the Flask development server with:
    - Host: 0.0.0.0 (accessible from all network interfaces)
    - Port: 5111 (configurable via environment)
    - Debug: True (enables hot reload and detailed error pages)
    
    For production deployment, use a WSGI server like Gunicorn:
        gunicorn -w 4 -b 0.0.0.0:5111 app:app
    """
    print("🌐 Starting ANN Savings Calculator Web Application...")
    print("📍 Access the application at: http://localhost:5111")
    print("🔧 Debug mode enabled for development")
    
    app.run(
        host="0.0.0.0",    # Accept connections from any IP
        port=5111,         # Application port
        debug=True         # Enable debug mode for development
    )

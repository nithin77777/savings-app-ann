
"""
ANN Savings Prediction Model Training Script

This script implements a deep learning approach to predict financial behavior
using an Artificial Neural Network (ANN). The model learns to predict:
1. Spending percentage based on income
2. Saving percentage based on financial patterns  
3. Goal achievement probability

The trained model is used by the Flask web application to provide
real-time financial predictions and personalized recommendations.

Architecture:
- Input Layer: 3 features (Income, Spending, Savings Goal)
- Hidden Layer 1: 16 neurons with ReLU activation
- Hidden Layer 2: 8 neurons with ReLU activation
- Output Layer: 3 outputs with sigmoid activation

Author: ANN Savings Calculator Team
Created: 2025
Based on: Collaborative research and development
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib


def load_and_prepare_data():
    """
    Load and prepare the synthetic financial dataset for training.
    
    Returns:
    --------
    tuple: (X_train_scaled, X_test_scaled, y_train, y_test, scaler)
        Prepared training and testing data with fitted scaler
    """
    print("📂 Loading dataset...")
    
    # Load the generated synthetic data
    df = pd.read_csv('./data/raw_data.csv')
    print(f"📊 Dataset shape: {df.shape}")
    print(f"🔍 Dataset preview:\n{df.head()}")
    
    # Define features (X) and targets (y)
    # Features: Income, Spending, Saving_Goal
    X = df[['Income', 'Spending', 'Saving_Goal']].copy()
    
    # Targets: Spending_Percent, Saving_Percent, Goal_Met
    y = df[['Spending_Percent', 'Saving_Percent', 'Goal_Met']].copy()
    
    print(f"✅ Features (X): {list(X.columns)}")
    print(f"✅ Targets (y): {list(y.columns)}")
    
    # Split data into training and testing sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42
    )
    
    print(f"🔄 Train samples: {X_train.shape[0]}")
    print(f"🔄 Test samples: {X_test.shape[0]}")
    
    # Feature scaling using MinMaxScaler (scales to 0-1 range)
    # This is crucial for neural network performance
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("⚖️  Feature scaling completed")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def build_model():
    """
    Build and compile the ANN model architecture.
    
    Architecture Design:
    - Sequential model for straightforward layer stacking
    - Dense layers with ReLU activation for non-linearity
    - Sigmoid output for bounded predictions (0-1 range)
    - Adam optimizer for adaptive learning rates
    - MSE loss for regression-style multi-output prediction
    
    Returns:
    --------
    tf.keras.Model: Compiled neural network model
    """
    print("🧠 Building ANN model architecture...")
    
    model = models.Sequential([
        # Input layer + First hidden layer
        # 16 neurons, ReLU activation, bias enabled
        layers.Dense(
            16, 
            activation='relu', 
            input_shape=(3,),  # 3 input features
            use_bias=True,
            name='hidden_layer_1'
        ),
        
        # Second hidden layer  
        # 8 neurons, ReLU activation for pattern recognition
        layers.Dense(
            8, 
            activation='relu',
            name='hidden_layer_2'
        ),
        
        # Output layer
        # 3 outputs: spending%, saving%, goal_achievement
        # Sigmoid activation ensures outputs are in [0,1] range
        layers.Dense(
            3, 
            activation='sigmoid',
            name='output_layer'
        )
    ])
    
    # Compile model with optimizer, loss, and metrics
    model.compile(
        optimizer='adam',          # Adaptive learning rate optimizer
        loss='mse',               # Mean Squared Error for multi-output regression
        metrics=['accuracy', 'mae'] # Track accuracy and Mean Absolute Error
    )
    
    print("✅ Model architecture:")
    model.summary()
    
    return model


def train_model(model, X_train_scaled, y_train):
    """
    Train the neural network model with the prepared data.
    
    Training Configuration:
    - 50 epochs for sufficient convergence
    - Batch size of 32 for stable gradient updates
    - 20% validation split for monitoring overfitting
    
    Parameters:
    -----------
    model : tf.keras.Model
        Compiled neural network model
    X_train_scaled : numpy.ndarray
        Scaled training features
    y_train : pandas.DataFrame
        Training targets
        
    Returns:
    --------
    tf.keras.callbacks.History: Training history with metrics
    """
    print("🚀 Starting model training...")
    
    # Train the model
    history = model.fit(
        X_train_scaled, y_train,
        epochs=50,              # Number of complete passes through data
        batch_size=32,          # Samples per gradient update
        validation_split=0.2,   # Use 20% of training data for validation
        verbose=1               # Show progress bar
    )
    
    print("✅ Training completed!")
    
    return history


def evaluate_model(model, X_test_scaled, y_test):
    """
    Evaluate the trained model on test data.
    
    Parameters:
    -----------
    model : tf.keras.Model
        Trained neural network model
    X_test_scaled : numpy.ndarray
        Scaled test features
    y_test : pandas.DataFrame
        Test targets
    """
    print("📊 Evaluating model performance...")
    
    # Evaluate model on test set
    loss, accuracy, mae = model.evaluate(X_test_scaled, y_test, verbose=0)
    
    print(f"📈 Test Results:")
    print(f"   Loss: {loss:.4f}")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   MAE: {mae:.4f}")
    
    return loss, accuracy, mae


def save_model_artifacts(model, scaler):
    """
    Save the trained model and scaler for production use.
    
    Parameters:
    -----------
    model : tf.keras.Model
        Trained neural network model
    scaler : sklearn.preprocessing.MinMaxScaler
        Fitted feature scaler
    """
    print("💾 Saving model artifacts...")
    
    # Save the trained Keras model
    model.save('./savings_model.keras')
    print("✅ Model saved as ./savings_model.keras")
    
    # Save the fitted scaler using joblib
    joblib.dump(scaler, './scaler.pkl')
    print("✅ Scaler saved as ./scaler.pkl")
    
    print("\n🎉 Training pipeline completed successfully!")
    print("🌐 Model ready for deployment in Flask web application")


def main():
    """
    Main training pipeline execution.
    
    Pipeline Steps:
    1. Load and prepare data
    2. Build model architecture  
    3. Train the model
    4. Evaluate performance
    5. Save artifacts for production
    """
    print("=" * 60)
    print("🧠 ANN SAVINGS PREDICTION MODEL TRAINING")
    print("=" * 60)
    
    try:
        # Step 1: Data preparation
        X_train_scaled, X_test_scaled, y_train, y_test, scaler = load_and_prepare_data()
        
        # Step 2: Model building
        model = build_model()
        
        # Step 3: Model training
        history = train_model(model, X_train_scaled, y_train)
        
        # Step 4: Model evaluation
        loss, accuracy, mae = evaluate_model(model, X_test_scaled, y_test)
        
        # Step 5: Save artifacts
        save_model_artifacts(model, scaler)
        
        print("\n" + "=" * 60)
        print("🎊 TRAINING COMPLETED SUCCESSFULLY!")
        print(f"📊 Final Test MAE: {mae:.4f}")
        print("🌐 Ready for web application deployment!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        raise


# Script execution entry point
if __name__ == '__main__':
    """
    Execute the training pipeline when script is run directly.
    
    Usage:
        python train_model.py
        
    Prerequisites:
        - Raw data must exist in './data/raw_data.csv'
        - All required packages must be installed (see requirements.txt)
    """
    main()
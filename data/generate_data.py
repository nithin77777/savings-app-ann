"""
Synthetic Data Generator for ANN Savings Prediction Model

This script generates realistic synthetic financial data for training
the neural network model. The data simulates various income levels,
spending patterns, and savings goals to create a comprehensive
dataset for model training.

Author: ANN Savings Calculator Team
Created: 2025
"""

import numpy as np
import pandas as pd


def generate_data(n=20000, seed=42):
    """
    Generate synthetic financial data for training the ANN model.
    
    This function creates realistic financial scenarios with:
    - Random income levels between ₹10,000 - ₹1,00,000
    - Variable spending ratios (10% - 90% of income)
    - Diverse savings goals (10% - 90% target)
    - Calculated spending/saving percentages
    - Goal achievement classification
    
    Parameters:
    -----------
    n : int, default=20000
        Number of synthetic samples to generate
    seed : int, default=42
        Random seed for reproducible results
        
    Returns:
    --------
    None
        Saves generated data to './data/raw_data.csv'
    """
    # Set random seed for reproducible results
    np.random.seed(seed)
    
    # Initialize list to store generated samples
    data = []

    # Generate n samples of financial data
    for i in range(n):
        # Generate random monthly income (₹10,000 to ₹1,00,000)
        income = np.random.randint(10000, 100001)

        # Generate random spending ratio (10% to 90% of income)
        spending_ratio = np.random.uniform(0.1, 0.9)
        spending = income * spending_ratio

        # Generate random savings goal (10% to 90% target)
        saving_goal = np.random.uniform(0.1, 0.9)

        # Calculate actual spending percentage of income
        spending_percent = spending / income
        
        # Calculate actual saving percentage (remaining after spending)
        saving_percent = 1 - spending_percent

        # Determine if savings goal is achieved
        # Goal is met if actual saving percentage >= target goal
        if saving_percent >= saving_goal:
            goal_met = 1  # Goal achieved
        else:
            goal_met = 0  # Goal not achieved
        
        # Append sample to data list
        data.append([
            income, 
            spending, 
            saving_goal, 
            spending_percent, 
            saving_percent, 
            goal_met
        ])
    
    # Define column names for the dataset
    cols = [
        'Income',           # Monthly income in INR
        'Spending',         # Monthly spending in INR
        'Saving_Goal',      # Target savings percentage (0-1)
        'Spending_Percent', # Actual spending percentage (0-1)
        'Saving_Percent',   # Actual saving percentage (0-1)
        'Goal_Met'          # Binary: 1 if goal achieved, 0 otherwise
    ]
    
    # Create DataFrame from generated data
    df = pd.DataFrame(data, columns=cols)

    # Save the dataset to CSV file
    df.to_csv('./raw_data.csv', index=False)
    print("✅ Data generated successfully → data/raw_data.csv")
    print(f"📊 Generated {n} samples with {len(cols)} features")
    print(f"📈 Income range: ₹{df['Income'].min():,} - ₹{df['Income'].max():,}")
    print(f"🎯 Goal achievement rate: {df['Goal_Met'].mean():.2%}")


if __name__ == '__main__':
    """
    Main execution block - generates synthetic data when script is run directly.
    
    Usage:
        python generate_data.py
    """
    print("🚀 Starting synthetic data generation for ANN model...")
    generate_data()


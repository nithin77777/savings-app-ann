# ANN Savings Calculator

A web-based savings calculator that uses Artificial Neural Networks (ANN) to predict spending and saving patterns based on user input. The application helps users understand their financial behavior and provides personalized recommendations for achieving their savings goals.

## 🎯 Features

- **Neural Network Predictions**: Uses TensorFlow/Keras ANN model to predict spending and saving percentages
- **Goal Achievement Analysis**: Determines if users can meet their savings goals
- **Interactive Web Interface**: Clean, responsive Flask web application
- **Visual Analytics**: Dynamic charts showing spending vs saving patterns
- **Personalized Recommendations**: AI-driven suggestions for financial improvement
- **Reward System**: Motivational rewards for goal achievement

## 🏗️ Project Structure

```
ANN/
├── data/
│   ├── generate_data.py      # Synthetic data generation for training
│   ├── raw_data.csv          # Generated training dataset
│   └── model_raw_data.csv    # Model-specific data
├── models/
│   ├── train_model.py        # Neural network training script
│   ├── evaluate.py           # Model evaluation and metrics
│   ├── savings_model.keras   # Trained Keras model
│   ├── savings_modela.h5     # Alternative model format
│   ├── scaler.pkl           # Feature scaling transformer
│   └── extra_metrics.json   # Additional evaluation metrics
├── webapp/
│   ├── app.py               # Flask application main entry point
│   ├── templates/
│   │   └── index.html       # Web interface template
│   └── static/
│       ├── script.js        # Frontend JavaScript logic
│       └── style.css        # Application styling
├── requirements.txt         # Python dependencies
└── README.md               # Project documentation
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd ANN
   ```

2. **Create virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Generate training data**

   ```bash
   cd data
   python generate_data.py
   cd ..
   ```

5. **Train the model**

   ```bash
   cd models
   python train_model.py
   cd ..
   ```

6. **Run the web application**

   ```bash
   cd webapp
   python app.py
   ```

7. **Access the application**
   Open your browser and navigate to `http://localhost:5111`

## 🧠 Model Architecture

The ANN model uses a sequential architecture with:

- **Input Layer**: 3 features (Income, Spending, Savings Goal)
- **Hidden Layer 1**: 16 neurons with ReLU activation
- **Hidden Layer 2**: 8 neurons with ReLU activation
- **Output Layer**: 3 outputs with sigmoid activation
  - Spending Percentage
  - Saving Percentage
  - Goal Achievement Score

### Training Details

- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error (MSE)
- **Metrics**: Accuracy, Mean Absolute Error (MAE)
- **Epochs**: 50
- **Batch Size**: 32
- **Validation Split**: 20%

## 📊 Data Generation

The synthetic dataset includes 20,000 samples with:

- **Income**: Random values between ₹10,000 - ₹100,000
- **Spending**: Variable percentage of income (10% - 90%)
- **Savings Goal**: Target savings percentage (10% - 90%)
- **Calculated Fields**: Spending percentage, saving percentage, goal achievement

## 🌐 Web Application

### API Endpoints

- **GET /** - Main application interface
- **POST /api/predict** - Prediction endpoint

### Input Parameters

- `income`: Monthly income in INR
- `spending`: Monthly spending in INR
- `savings_goal`: Target savings percentage (1-90%)

### Response Format

```json
{
	"ok": true,
	"result": {
		"spending_percentage": 60.0,
		"savings_percentage": 40.0,
		"goal_achieved": true,
		"recommendation": "You're on track for your 40% savings goal.",
		"note": "Congrats you have achieved your goal - Your Reward is \"20.0 INR\"..."
	}
}
```

## 📈 Model Performance

The trained model achieves:

- **MAE (Spending)**: Low prediction error for spending patterns
- **MAE (Saving)**: Accurate saving percentage predictions
- **R² Score**: High correlation with actual values
- **Goal Accuracy**: Precise goal achievement classification

## 🛠️ Technologies Used

- **Backend**: Flask, Python
- **Machine Learning**: TensorFlow/Keras, Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Frontend**: HTML5, CSS3, JavaScript
- **Visualization**: Chart.js
- **Model Persistence**: Joblib

## 🎨 Features in Detail

### Intelligent Recommendations

- Analyzes spending patterns using neural networks
- Provides actionable advice for goal achievement
- Calculates required spending reductions

### Reward System

- Motivational rewards based on goal achievement
- Reward amount calculated as: `Goal% × 0.001 × Income`
- Encourages consistent saving behavior

### Visual Analytics

- Interactive bar charts comparing spending, saving, and goals
- Real-time updates based on user input
- Responsive design for all devices

## 🔧 Configuration

Environment variables:

- `MODEL_PATH`: Path to the Keras model file (default: `models/savings_model.keras`)
- `SCALER_PATH`: Path to the scaler file (default: `models/scaler.pkl`)

## 📝 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Support

For questions or support, please open an issue in the GitHub repository.

---

_Built with ❤️ using Neural Networks and Flask_

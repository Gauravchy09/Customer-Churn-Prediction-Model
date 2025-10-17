# Customer Churn Prediction Using Artificial Neural Networks (ANN)

## Project Overview
This project predicts customer churn using an Artificial Neural Network (ANN). Churn prediction helps businesses identify customers who are likely to leave, enabling proactive retention strategies and improving overall customer satisfaction.

The model is trained on historical customer data and predicts whether a customer will churn or not based on various features such as demographics, account details, and usage behavior.

## Features Used
- Customer demographics (e.g., age, gender, location)  
- Account details (e.g., tenure, subscription type)  
- Usage patterns (e.g., service usage metrics, interactions)  
- Other relevant customer behavior indicators  

## Technologies and Tools
- Python 3.x  
- TensorFlow / Keras  
- NumPy & Pandas  
- Scikit-learn  
- Matplotlib / Seaborn (for data visualization)  
- Pickle (for saving models and encoders)  

## Installation
1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd <repo-folder>
   
Create and activate a virtual environment:

python -m venv venv
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate


Install dependencies:

pip install -r requirements.txt

Usage

Load and preprocess the dataset.

Train the ANN model:

python train_model.py


Make predictions on new customer data:

python predict_churn.py

Model Performance

Accuracy: 80.33%

Future Improvements

Hyperparameter tuning using GridSearch or Bayesian optimization

Deploy the model as a web application or REST API

Incorporate more customer behavior features for better accuracy

# Machine Learning Projects Collection

This repository contains three machine learning projects focused on healthcare, sentiment analysis, and customer retention analysis.

## Project Dependencies

### 1. Diabetes Prediction Project
```bash
pip install tensorflow
pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn
pip install plotly
pip install ipython
pip install adabelief-tf
pip install mlxtend
pip install scikit-learn
```

### 2. Customer Churn Analysis Project
```bash
pip install numpy
pip install pandas
pip install seaborn
pip install matplotlib
pip install graphviz
pip install scikit-learn
pip install mlxtend
pip install folium
pip install plotly
pip install hyperopt
pip install xgboost
pip install ipython
```

### 3. Sentiment Analysis Project
```bash
pip install numpy
pip install pandas
pip install matplotlib
pip install seaborn
pip install tensorflow
pip install tensorflow-hub
pip install scikit-learn
pip install scikit-plot
```

## 1. Diabetes Prediction using Artificial Neural Networks

### Overview
This notebook implements a deep learning model to predict diabetes in patients using an Artificial Neural Network (ANN) architecture.

### Key Features
- Utilizes neural networks for binary classification, whether the patient has diabetes or not.
- Preprocesses medical data for optimal model performance
- Includes data normalization and feature scaling
- Implements early stopping and model optimization techniques

### Key Libraries Used
- TensorFlow/Keras for neural network implementation
- AdaBelief optimizer for model optimization
- Plotly and Seaborn for advanced visualizations
- MLXtend for preprocessing

## 2. Company Review Sentiment Analysis

### Overview
This project analyzes customer reviews using Google's Universal Sentence Encoder (USE) to determine sentiment polarity.

### Key Features
- Implements Universal Sentence Encoder for text embedding
- Performs sentiment classification on Likert Scale from 1 to 5
- Handles text preprocessing and cleaning
- Provides sentiment score analysis

### Key Libraries Used
- TensorFlow Hub for Universal Sentence Encoder
- TensorFlow/Keras for model building
- Scikit-learn for metrics and evaluation
- Scikit-plot for ROC curve visualization

## 3. Customer Churn Prediction

### Overview
A comparative analysis of three different machine learning models to predict customer churn.

### Models Implemented
1. Logistic Regression
2. Random Forest
3. Decision Tree

### Key Libraries Used
- Scikit-learn for model implementation
- XGBoost for gradient boosting
- Hyperopt for hyperparameter optimization
- Plotly for interactive visualizations
- Folium for geographical visualizations

## Setup and Installation

### Environment Setup
```bash
# Create and activate virtual environment (recommended)
python -m venv mlenv
source mlenv/bin/activate  # On Windows: mlenv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

### requirements.txt
```
tensorflow>=2.0.0
pandas
numpy
matplotlib
seaborn
plotly
ipython
adabelief-tf
mlxtend
scikit-learn
graphviz
folium
hyperopt
xgboost
tensorflow-hub
scikit-plot
```

## Usage
Each notebook can be run independently. Make sure to:
1. Install all required dependencies
2. Prepare the corresponding dataset
3. Follow the notebook instructions step by step

## Model Performance

Each project includes:
- Training and validation metrics
- Model evaluation scores
- Performance visualization
- Confusion matrices where applicable

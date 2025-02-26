# Vegetable Price Prediction System

This repository contains a machine learning project aimed at predicting the prices of various vegetables. Accurate price forecasting can assist farmers, retailers, and consumers in making informed decisions.

## Project Overview

The Vegetable Price Prediction System utilizes historical price data and relevant features to forecast future vegetable prices. The project is implemented in Python and leverages machine learning techniques for predictive modeling.

## Features

- **Data Preprocessing**: Cleansing and preparing raw data for analysis.
- **Exploratory Data Analysis (EDA)**: Visualizing data to uncover patterns and insights.
- **Model Training**: Implementing machine learning algorithms to train predictive models.
- **Prediction Interface**: A user-friendly interface to input data and obtain price predictions.

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/TANJUMAJERIN/vegetable-price-prediction-system.git
   ```


2. **Navigate to the Project Directory**:

   ```bash
   cd vegetable-price-prediction-system
   ```


3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```


## Usage

1. **Data Preparation**:

   - Place your dataset (`data.csv`) in the project directory.
   - Run the data preprocessing script:

     ```bash
     python data_preprocessing.py
     ```

2. **Train the Model**:

   - Execute the training script:

     ```bash
     python final.py
     ```

   - This will train the model and save it for future predictions.

3. **Make Predictions**:

   - Run the prediction interface:

     ```bash
     python app.py
     ```

   - Open your web browser and navigate to `http://localhost:5000` to access the interface.

## Dependencies

- Python 3.x
- Flask
- Pandas
- Scikit-learn
- NumPy
- Matplotlib

Ensure all dependencies are installed by running:


```bash
pip install -r requirements.txt
```

For any questions or support, please contact bsse1312@iit.du.ac.bd 

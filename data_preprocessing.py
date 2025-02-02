import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def preprocess_data(file_path):
    """
    Read and preprocess data from CSV file
    """
    try:
        # Read CSV file
        data = pd.read_csv(file_path)
        data.columns = data.columns.str.strip() 
       
        data['DATE'] = pd.to_datetime(data['DATE'])
        
        
        data['day_of_year'] = data['DATE'].dt.dayofyear
        data['month'] = data['DATE'].dt.month
        
       
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data[numeric_columns] = data[numeric_columns].fillna(method='ffill')
        
       
        required_columns = ['DATE', 'Natural_disaster']
        vegetable_columns = ['Potato', 'Onion', 'Garlic', 'Chili', 'Ginger', 'Egg']
        
        for veg in vegetable_columns:
            min_col = f'{veg}_min'
            max_col = f'{veg}_max'
            if min_col not in data.columns or max_col not in data.columns:
                raise ValueError(f"Missing required columns for {veg}")
        
        return data
        
    except FileNotFoundError:
        raise FileNotFoundError("Data file not found. Please check the file path.")
    except Exception as e:
        raise Exception(f"Error processing data: {str(e)}")

def create_model(vegetable, data):
    """
    Create and train Random Forest model for both min and max prices
    """
  
    features = ['day_of_year', 'month', 'Natural_disaster']
    
 
    y_min = data[f'{vegetable}_min']
    y_max = data[f'{vegetable}_max']
    
    
    X_train, X_test, y_train_min, y_test_min = train_test_split(data[features], y_min, test_size=0.2, random_state=42)
    _, _, y_train_max, y_test_max = train_test_split(data[features], y_max, test_size=0.2, random_state=42)
    
   
    model_min = RandomForestRegressor(n_estimators=100, random_state=42)
    model_max = RandomForestRegressor(n_estimators=100, random_state=42)
    
  
    model_min.fit(X_train, y_train_min)
    model_max.fit(X_train, y_train_max)
    
    
    train_score_min = model_min.score(X_train, y_train_min)
    test_score_min = model_min.score(X_test, y_test_min)
    
    train_score_max = model_max.score(X_train, y_train_max)
    test_score_max = model_max.score(X_test, y_test_max)
    
    return model_min, model_max, train_score_min, test_score_min, train_score_max, test_score_max


def predict_price(model_min, model_max, date, natural_disaster):
    """
    Make price prediction for given date and conditions for both min and max prices
    """
    
    date = pd.to_datetime(date)
    features = np.array([[date.dayofyear, date.month, natural_disaster]])
    
  
    predicted_min_price = model_min.predict(features)[0]
    predicted_max_price = model_max.predict(features)[0]
    
    return predicted_min_price, predicted_max_price



# import pandas as pd
# import numpy as np
# from datetime import datetime
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split

# def preprocess_data(file_path):
#     """
#     Read and preprocess data from CSV file
#     """
#     try:
#         # Read CSV file
#         data = pd.read_csv(file_path)
#         data.columns = data.columns.str.strip()  # Clean any extra spaces
#         #st.write(data.columns)  # Display column names in Streamlit

        
#         # Convert DATE to datetime
#         data['DATE'] = pd.to_datetime(data['DATE'])
        
#         # Add numerical features for date
#         data['day_of_year'] = data['DATE'].dt.dayofyear
#         data['month'] = data['DATE'].dt.month
        
#         # Handle missing values if any
#         numeric_columns = data.select_dtypes(include=[np.number]).columns
#         data[numeric_columns] = data[numeric_columns].fillna(method='ffill')
        
#         # Ensure all required columns exist
#         required_columns = ['DATE', 'Natural_disaster']
#         vegetable_columns = ['Potato', 'Onion', 'Garlic', 'Chili', 'Ginger', 'Egg']
        
#         for veg in vegetable_columns:
#             min_col = f'{veg}_min'
#             max_col = f'{veg}_max'
#             if min_col not in data.columns or max_col not in data.columns:
#                 raise ValueError(f"Missing required columns for {veg}")
        
#         return data
        
#     except FileNotFoundError:
#         raise FileNotFoundError("Data file not found. Please check the file path.")
#     except Exception as e:
#         raise Exception(f"Error processing data: {str(e)}")
# def create_model(vegetable, data):
#     """
#     Create and train polynomial regression model for both min and max prices
#     """
#     # Select features
#     features = ['day_of_year', 'month', 'Natural_disaster']
    
#     # Train models for both min and max prices
#     y_min = data[f'{vegetable}_min']
#     y_max = data[f'{vegetable}_max']
    
#     # Split data into train and test sets
#     X_train, X_test, y_train_min, y_test_min = train_test_split(data[features], y_min, test_size=0.2, random_state=42)
#     _, _, y_train_max, y_test_max = train_test_split(data[features], y_max, test_size=0.2, random_state=42)
    
#     # Create polynomial features
#     poly = PolynomialFeatures(degree=2)
#     X_train_poly = poly.fit_transform(X_train)
#     X_test_poly = poly.transform(X_test)
    
#     # Train models for min and max prices
#     model_min = LinearRegression()
#     model_max = LinearRegression()
    
#     model_min.fit(X_train_poly, y_train_min)
#     model_max.fit(X_train_poly, y_train_max)
    
#     # Calculate model scores
#     train_score_min = model_min.score(X_train_poly, y_train_min)
#     test_score_min = model_min.score(X_test_poly, y_test_min)
    
#     train_score_max = model_max.score(X_train_poly, y_train_max)
#     test_score_max = model_max.score(X_test_poly, y_test_max)
    
#     return model_min, model_max, poly, train_score_min, test_score_min, train_score_max, test_score_max


# def predict_price(model_min, model_max, poly, date, natural_disaster):
#     """
#     Make price prediction for given date and conditions for both min and max prices
#     """
#     # Create feature vector
#     date = pd.to_datetime(date)
#     features = np.array([[date.dayofyear, date.month, natural_disaster]])
#     features_poly = poly.transform(features)
    
#     # Make predictions for both min and max prices
#     predicted_min_price = model_min.predict(features_poly)[0]
#     predicted_max_price = model_max.predict(features_poly)[0]
    
#     return predicted_min_price, predicted_max_price
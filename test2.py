import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from datetime import datetime


df = pd.read_csv('data.csv')
df['DATE'] = pd.to_datetime(df['DATE'])


vegetable_mapping = {
    'Potato': ('Potato', 'potato.jpg'),
    'Onion': ('Onion', 'onion.jpg'),
    'Garlic': ('Garlic', 'garlic.jpg'),
    'Chili': ('Chili', 'chilli.jpg'),
    'Ginger': ('Ginger', 'ginger.jpg'),
    'Egg': ('Egg', 'egg.jpg')
}


def get_recent_price(veg_name):
    latest_row = df.sort_values(by='DATE', ascending=False).iloc[0]
    return latest_row[f'{veg_name}_min'], latest_row[f'{veg_name}_max']


def train_model(vegetable, n_estimators=100):
    veg_name = vegetable_mapping[vegetable][0]
    X = df[['DATE', 'Natural_disaster', 'Polytical_issue']].copy()
    X['DATE'] = X['DATE'].map(datetime.toordinal)
    y_min = df[f'{veg_name}_min']
    y_max = df[f'{veg_name}_max']
    
    model_min = RandomForestRegressor(n_estimators=n_estimators, random_state=42).fit(X, y_min)
    model_max = RandomForestRegressor(n_estimators=n_estimators, random_state=42).fit(X, y_max)
    
    return model_min, model_max


def predict_price(vegetable, date, disaster, issue, model_min, model_max):
    date_ordinal = datetime.strptime(date, '%Y-%m-%d').toordinal()
    X_input = pd.DataFrame([[date_ordinal, disaster, issue]], columns=['DATE', 'Natural_disaster', 'Polytical_issue'])
    min_price = model_min.predict(X_input)[0]
    max_price = model_max.predict(X_input)[0]
    return round(min_price, 2), round(max_price, 2)


def calculate_accuracy(vegetable, model_min, model_max):
    veg_name = vegetable_mapping[vegetable][0]
    X = df[['DATE', 'Natural_disaster', 'Polytical_issue']].copy()
    X['DATE'] = X['DATE'].map(datetime.toordinal)
    
    y_min = df[f'{veg_name}_min']
    y_max = df[f'{veg_name}_max']
    
   
    predicted_min = model_min.predict(X)
    predicted_max = model_max.predict(X)
    
 
    accuracy_min = np.mean(np.abs((predicted_min - y_min) / y_min) < 0.1) * 100  
    accuracy_max = np.mean(np.abs((predicted_max - y_max) / y_max) < 0.1) * 100  
    
    return accuracy_min, accuracy_max


st.set_page_config(page_title="Vegetable Price Predictor", layout="wide")
st.title('Vegetable Price Predictor')


st.image('vegetables.jpg', use_container_width=True)  


st.subheader("Predict Future Prices")
vegetable = st.selectbox("Select a vegetable", list(vegetable_mapping.keys()))
date = st.date_input("Select a date", datetime.today()).strftime('%Y-%m-%d')
disaster = st.checkbox("Natural Disaster?")
issue = st.checkbox("Political Issue?")

disaster = 1 if disaster else 0
issue = 1 if issue else 0

# Train model
model_min, model_max = train_model(vegetable)

if st.button("Predict Price"):
    min_price, max_price = predict_price(vegetable, date, disaster, issue, model_min, model_max)
    st.success(f"Predicted {vegetable} Price on {date}: {min_price} - {max_price} BDT")
if st.button("Show Price Trend"):
    veg_name = vegetable_mapping[vegetable][0]
    df_filtered = df[['DATE', f'{veg_name}_min', f'{veg_name}_max']]
    df_filtered.set_index('DATE', inplace=True)
    
    plt.figure(figsize=(10, 5))
    plt.plot(df_filtered.index, df_filtered[f'{veg_name}_min'], label='Min Price', marker='o')
    plt.plot(df_filtered.index, df_filtered[f'{veg_name}_max'], label='Max Price', marker='o')
    plt.xlabel('Date')
    plt.ylabel('Price (BDT)')
    plt.title(f'{veg_name} Price Trend')
    plt.legend()
    st.pyplot(plt)

st.subheader("Recent Prices")

today_date = datetime.today().strftime('%Y-%m-%d')


st.subheader("Vegetable Information")

cols = st.columns(len(vegetable_mapping))
for col, (veg, (veg_name, img)) in zip(cols, vegetable_mapping.items()):
    min_price, max_price = get_recent_price(veg_name)
    with col:
        st.image(img, use_container_width=True)  
        st.markdown(f"### {veg}")
        st.markdown(f"💰 **{min_price} - {max_price}** BDT")


if st.button("Show Model Accuracy"):
    accuracy_min, accuracy_max = calculate_accuracy(vegetable, model_min, model_max)
    st.success(f"Model Accuracy for {vegetable}:\n"
               f"Min Price Accuracy: {accuracy_min:.2f}%\n"
               f"Max Price Accuracy: {accuracy_max:.2f}%")
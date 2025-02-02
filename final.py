import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

# Set full-screen layout
st.set_page_config(page_title="Vegetable Price Predictor", layout="wide")

# Load dataset
df = pd.read_csv('data.csv')
df['DATE'] = pd.to_datetime(df['DATE'])

# Mapping UI-friendly names to dataset column names
vegetable_mapping = {
    'Potato 🥔': 'Potato',
    'Onion 🧅': 'Onion',
    'Garlic 🧄': 'Garlic',
    'Chili 🌶️': 'Chili',
    'Ginger 🫚': 'Ginger',
    'Egg 🥚': 'Egg'
}

# Function to train Random Forest models
def train_model(vegetable_name, n_estimators=100):
    # Remove emojis and map to actual vegetable name
    veg_name = vegetable_mapping.get(vegetable_name)
    if veg_name is None:
        raise ValueError(f"Vegetable {vegetable_name} not found in mapping.")
    
    X = df[['DATE', 'Natural_disaster', 'Polytical_issue']].copy()
    X['DATE'] = X['DATE'].map(datetime.toordinal)
    y_min = df[f'{veg_name}_min']
    y_max = df[f'{veg_name}_max']
    
    model_min = RandomForestRegressor(n_estimators=n_estimators, random_state=42).fit(X, y_min)
    model_max = RandomForestRegressor(n_estimators=n_estimators, random_state=42).fit(X, y_max)
    
    return model_min, model_max

# Function to predict prices
def predict_price(vegetable, date, disaster, issue, model_min, model_max):
    date_ordinal = datetime.strptime(date, '%Y-%m-%d').toordinal()
    X_input = pd.DataFrame([[date_ordinal, disaster, issue]], columns=['DATE', 'Natural_disaster', 'Polytical_issue'])
    
    min_price = model_min.predict(X_input)[0]
    max_price = model_max.predict(X_input)[0]
    return round(min_price, 2), round(max_price, 2)

# Custom styling
st.markdown(
    """
    <style>
        .title-container {
            text-align: center;
            font-size: 32px;
            font-weight: bold;
            margin-bottom: 10px;
            color: #2c3e50;
        }
        .image-container {
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .price-box {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
            text-align: center;
            font-size: 18px;
        }
        .stButton>button {
            width: 100%;
            padding: 10px;
            font-size: 18px;
            background-color: #007bff;
            color: white;
            border-radius: 5px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Centered Title
st.markdown("<div class='title-container'>🌿 Vegetable Price Predictor</div>", unsafe_allow_html=True)

# Centered Image
st.markdown("<div class='image-container'>", unsafe_allow_html=True)
st.image("vegetable.jpg", use_column_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# Layout in columns
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("📅 Select Date & Factors")
    date = st.date_input("Select a date", datetime.today()).strftime('%Y-%m-%d')
    disaster = st.checkbox("Natural Disaster?") 
    issue = st.checkbox("Political Issue?")
    disaster = 1 if disaster else 0
    issue = 1 if issue else 0

with col2:
    st.subheader("🥦 Select a Vegetable")
    vegetable = st.selectbox("Choose a vegetable", list(vegetable_mapping.keys()))

# Train the model
model_min, model_max = train_model(vegetable)

col3, col4 = st.columns([1, 1])

with col3:
    if st.button("🔍 Predict Price"):
        min_price, max_price = predict_price(vegetable, date, disaster, issue, model_min, model_max)
        st.markdown(f"""
        <div class='price-box'>
            <h2>Predicted {vegetable} Price on {date}</h2>
            <h3>💰 {min_price} - {max_price} BDT</h3>
        </div>
        """, unsafe_allow_html=True)

with col4:
    if st.button("📈 Show Price Trend"):
        veg_name = vegetable_mapping[vegetable]
        df_filtered = df[['DATE', f'{veg_name}_min', f'{veg_name}_max']]
        df_filtered.set_index('DATE', inplace=True)

        plt.figure(figsize=(10, 5))
        plt.plot(df_filtered.index, df_filtered[f'{veg_name}_min'], label='Min Price', marker='o', linestyle='dashed')
        plt.plot(df_filtered.index, df_filtered[f'{veg_name}_max'], label='Max Price', marker='o', linestyle='solid', color='red')
        plt.xlabel('Date')
        plt.ylabel('Price (BDT)')
        plt.title(f'{veg_name} Price Trend')
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)

# Display today's vegetable prices
st.subheader(f"🛒 Today's Market Prices - {datetime.today().strftime('%Y-%m-%d')}")

def predict_today_prices(date, disaster, issue):
    today_prices = {}
    for veg_name in vegetable_mapping.keys():
        model_min, model_max = train_model(veg_name)
        min_price, max_price = predict_price(veg_name, date, disaster, issue, model_min, model_max)
        today_prices[veg_name] = (min_price, max_price)
    return today_prices

# Get today's prices
today_prices = predict_today_prices(datetime.today().strftime('%Y-%m-%d'), disaster, issue)

# Display the predicted prices for today
if today_prices:
    cols = st.columns(len(vegetable_mapping))
    for idx, (veg, prices) in enumerate(today_prices.items()):
        min_price, max_price = prices
        with cols[idx]:
            st.markdown(f"""
            <div class='price-box'>
                <h3>{veg} (Today)</h3>
                <p>Min Price: ৳{min_price}</p>
                <p>Max Price: ৳{max_price}</p>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("📝 **Project by DU IIT | AI Course**")




# import pandas as pd
# import numpy as np
# import streamlit as st
# import matplotlib.pyplot as plt
# from sklearn.ensemble import RandomForestRegressor
# from datetime import datetime

# # Set full-screen layout
# st.set_page_config(page_title="Vegetable Price Predictor", layout="wide")

# # Load dataset
# df = pd.read_csv('data.csv')
# df['DATE'] = pd.to_datetime(df['DATE'])

# # Mapping UI-friendly names to dataset column names
# vegetable_mapping = {
#     'Potato 🥔': 'Potato',
#     'Onion 🧅': 'Onion',
#     'Garlic 🧄': 'Garlic',
#     'Chili 🌶️': 'Chili',
#     'Ginger 🫚': 'Ginger',
#     'Egg 🥚': 'Egg'
# }

# # Function to train Random Forest models
# def train_model(vegetable, n_estimators=100):
#     veg_name = vegetable_mapping[vegetable]
#     X = df[['DATE', 'Natural_disaster', 'Polytical_issue']].copy()
#     X['DATE'] = X['DATE'].map(datetime.toordinal)
#     y_min = df[f'{veg_name}_min']
#     y_max = df[f'{veg_name}_max']
    
#     model_min = RandomForestRegressor(n_estimators=n_estimators, random_state=42).fit(X, y_min)
#     model_max = RandomForestRegressor(n_estimators=n_estimators, random_state=42).fit(X, y_max)
    
#     return model_min, model_max

# # Function to predict prices
# def predict_price(vegetable, date, disaster, issue, model_min, model_max):
#     date_ordinal = datetime.strptime(date, '%Y-%m-%d').toordinal()
#     X_input = pd.DataFrame([[date_ordinal, disaster, issue]], columns=['DATE', 'Natural_disaster', 'Polytical_issue'])
    
#     min_price = model_min.predict(X_input)[0]
#     max_price = model_max.predict(X_input)[0]
#     return round(min_price, 2), round(max_price, 2)

# # Custom styling
# st.markdown(
#     """
#     <style>
#         .title-container {
#             text-align: center;
#             font-size: 32px;
#             font-weight: bold;
#             margin-bottom: 10px;
#             color: #2c3e50;
#         }
#         .image-container {
#             display: flex;
#             justify-content: center;
#             align-items: center;
#         }
#         .price-box {
#             background-color: white;
#             padding: 20px;
#             border-radius: 10px;
#             box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
#             text-align: center;
#             font-size: 18px;
#         }
#         .stButton>button {
#             width: 100%;
#             padding: 10px;
#             font-size: 18px;
#             background-color: #007bff;
#             color: white;
#             border-radius: 5px;
#         }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # Centered Title
# st.markdown("<div class='title-container'>🌿 Vegetable Price Predictor</div>", unsafe_allow_html=True)

# # Centered Image
# st.markdown("<div class='image-container'>", unsafe_allow_html=True)
# st.image("vegetable.jpg", use_column_width=True)
# st.markdown("</div>", unsafe_allow_html=True)

# # Layout in columns
# col1, col2 = st.columns([3, 2])

# with col1:
#     st.subheader("📅 Select Date & Factors")
#     date = st.date_input("Select a date", datetime.today()).strftime('%Y-%m-%d')
#     disaster = st.checkbox("Natural Disaster?") 
#     issue = st.checkbox("Political Issue?")
#     disaster = 1 if disaster else 0
#     issue = 1 if issue else 0

# with col2:
#     st.subheader("🥦 Select a Vegetable")
#     vegetable = st.selectbox("Choose a vegetable", list(vegetable_mapping.keys()))

# # Train the model
# model_min, model_max = train_model(vegetable)

# col3, col4 = st.columns([1, 1])

# with col3:
#     if st.button("🔍 Predict Price"):
#         min_price, max_price = predict_price(vegetable, date, disaster, issue, model_min, model_max)
#         st.markdown(f"""
#         <div class='price-box'>
#             <h2>Predicted {vegetable} Price on {date}</h2>
#             <h3>💰 {min_price} - {max_price} BDT</h3>
#         </div>
#         """, unsafe_allow_html=True)

# with col4:
#     if st.button("📈 Show Price Trend"):
#         veg_name = vegetable_mapping[vegetable]
#         df_filtered = df[['DATE', f'{veg_name}_min', f'{veg_name}_max']]
#         df_filtered.set_index('DATE', inplace=True)

#         plt.figure(figsize=(10, 5))
#         plt.plot(df_filtered.index, df_filtered[f'{veg_name}_min'], label='Min Price', marker='o', linestyle='dashed')
#         plt.plot(df_filtered.index, df_filtered[f'{veg_name}_max'], label='Max Price', marker='o', linestyle='solid', color='red')
#         plt.xlabel('Date')
#         plt.ylabel('Price (BDT)')
#         plt.title(f'{veg_name} Price Trend')
#         plt.legend()
#         plt.grid(True)
#         st.pyplot(plt)

# # Display today's vegetable prices
# st.subheader(f"🛒 Today's Market Prices - {datetime.today().strftime('%Y-%m-%d')}")

# today_df = df[df['DATE'] == datetime.today().strftime('%Y-%m-%d')]

# if not today_df.empty:
#     cols = st.columns(len(vegetable_mapping))
#     for idx, (veg, veg_name) in enumerate(vegetable_mapping.items()):
#         min_price = today_df[f'{veg_name}_min'].values[0] if f'{veg_name}_min' in today_df else "N/A"
#         max_price = today_df[f'{veg_name}_max'].values[0] if f'{veg_name}_max' in today_df else "N/A"

#         with cols[idx]:
#             st.markdown(f"""
#             <div class='price-box'>
#                 <h3>{veg} (Today)</h3>
#                 <p>Min Price: ৳{min_price}</p>
#                 <p>Max Price: ৳{max_price}</p>
#             </div>
#             """, unsafe_allow_html=True)
# else:
#     st.write("No data available for today's market prices.")


# st.markdown("---")
# st.markdown("📝 **Project by DU IIT | AI Course**")



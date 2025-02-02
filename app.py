import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import os
from data_preprocessing import preprocess_data, create_model, predict_price

def set_custom_style():
    st.markdown("""
        <style>
        .css-1d391kg {
            width: 250px;
        }
        
        .stImage {
            width: 10%;
            height: auto; 
            display: block; 
            margin-left: auto; 
            margin-right: auto; 
        }

        .price-card {
            background-color: #A8E6A3;
            padding: 15px;
            border-radius: 8px;
            color: white;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            margin: 10px 0;
        }

                
        .footer {
            width: 100%;
            padding: 20px;
            text-align: center;
            border-top: 1px solid #ddd;
            display: flex;
            justify-content: center;
            align-items: center;
            margin-top: 50px;
        }

        .footer-text {
            margin: 0;
        }
        
        .footer-subtext {
            margin: 5px 0 0 0;
            font-size: 0.8em;
            color: #666;
        }

        .sidebar .sidebar-content {
            width: 250px;
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(layout="wide", page_title="Bangladesh Vegetable Market Prediction")
    set_custom_style()
    
    try:
        data_path = "data.csv"
        if not os.path.exists(data_path):
            st.error("data.csv not found in the current directory!")
            return
            
        data = preprocess_data(data_path)
        page = st.sidebar.selectbox("Select Page", ["Home", "Data Analysis"])
        
        if page == "Home":
            show_home_page(data)
        else:
            show_analysis_page(data)
            
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")

def show_home_page(data):
    st.image("vegetable.jpg", caption="Fresh Vegetables Market", use_column_width=True)
    st.title("Vegetable Price Prediction System")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        vegetable = st.selectbox("Select Vegetable", ["Potato", "Onion", "Garlic", "Chili", "Ginger", "Egg"])
    
    with col2:
        prediction_date = st.date_input("Select Date")
    
    with col3:
        natural_disaster = st.checkbox("Natural Disaster")
        political_issue = st.checkbox("Political Issue")
    
    if st.button("Predict Price"):
        model_min, model_max, train_score_min, test_score_min, train_score_max, test_score_max = create_model(vegetable, data)
        predicted_min_price, predicted_max_price = predict_price(
            model_min, model_max, prediction_date, int(natural_disaster)
        )
        
        st.success(f"Predicted price range for {vegetable}: Min: ৳{predicted_min_price:.2f}, Max: ৳{predicted_max_price:.2f}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model Accuracy (Training Min)", f"{train_score_min:.2%}")
        with col2:
            st.metric("Model Accuracy (Testing Min)", f"{test_score_min:.2%}")
        with col1:
            st.metric("Model Accuracy (Training Max)", f"{train_score_max:.2%}")
        with col2:
            st.metric("Model Accuracy (Testing Max)", f"{test_score_max:.2%}")
    
    st.subheader("Current Market Prices")
    latest_data = data.iloc[-1]
    
    items = [
        ("Potato", "🥔"), ("Onion", "🧅"), ("Garlic", "🧄"),
        ("Chili", "🌶️"), ("Ginger", "🫑"), ("Egg", "🥚")
    ]
    
    cols = st.columns(3)
    for idx, (item, emoji) in enumerate(items):
        with cols[idx // 2]:
            st.markdown(
                f"""
                <div class="price-card">
                    <h3 class="veggie-title">{emoji} {item}</h3>
                    <p class="price-text">Min: ৳{latest_data[f'{item}_min']}</p>
                    <p class="price-text">Max: ৳{latest_data[f'{item}_max']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    st.markdown(
        """
        <div class="footer">
            <p class="footer-text">© 2024 Bangladesh Vegetable Market Prediction System</p>
            
        </div>
        """,
        unsafe_allow_html=True
    )


def show_analysis_page(data):
    st.title("Market Data Analysis")
    
    vegetable = st.selectbox("Select Vegetable for Analysis", ["Potato", "Onion", "Garlic", "Chili", "Ginger", "Egg"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.line(
            data,
            x="DATE",
            y=[f"{vegetable}_min", f"{vegetable}_max"],
            title=f"{vegetable} Price Trends",
            labels={"value": "Price (৳)", "variable": "Price Type"},
            template="plotly_white"
        )
        fig.update_layout(legend_title_text="Price Range", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # FIXED: Corrected the unpacking of values from create_model()
        model_min, model_max, train_score_min, test_score_min, train_score_max, test_score_max = create_model(vegetable, data)

        data[f'{vegetable}_predicted'] = model_min.predict(
            data[['day_of_year', 'month', 'Natural_disaster']]
        )
        
        fig2 = px.line(
            data,
            x="DATE",
            y=[f"{vegetable}_max", f"{vegetable}_predicted"],
            title=f"{vegetable} Actual vs Predicted Prices",
            labels={"value": "Price (৳)", "variable": "Price Type"},
            template="plotly_white"
        )
        fig2.update_layout(legend_title_text="Price Type", hovermode="x unified")
        st.plotly_chart(fig2, use_container_width=True)
    
    st.subheader("Price Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_price = data[f"{vegetable}_max"].mean()
        st.metric("Average Price", f"৳{avg_price:.2f}")
    
    with col2:
        max_price = data[f"{vegetable}_max"].max()
        st.metric("Highest Price", f"৳{max_price:.2f}")
    
    with col3:
        min_price = data[f"{vegetable}_min"].min()
        st.metric("Lowest Price", f"৳{min_price:.2f}")
    
    st.subheader("Historical Data")
    st.dataframe(
        data[['DATE', f'{vegetable}_min', f'{vegetable}_max', 'Natural_disaster']]
        .style.highlight_max(axis=0, subset=[f'{vegetable}_max']),
        use_container_width=True
    )
    
if __name__ == "__main__":
    main()


# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# from datetime import datetime
# import os
# from data_preprocessing import preprocess_data, create_model, predict_price
# def set_custom_style():
#     st.markdown("""
#         <style>
#         /* Modern CSS without deprecated properties */
#         @media (forced-colors: active) {
#             .stApp {
#                 background-color: Canvas;
#                 color: CanvasText;
#             }
#         }
        
#         .veggie-card {
#             padding: 20px;
#             border-radius: 10px;
#             border: 1px solid #ddd;
#             margin: 10px;
#             background-color: white;
#             box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
#             transition: transform 0.2s;
#         }
        
#         .veggie-card:hover {
#             transform: translateY(-2px);
#         }
        
#         .veggie-title {
#             color: #1f77b4;
#             margin-bottom: 15px;
#         }
        
#         .price-text {
#             font-size: 1.2em;
#             margin: 5px 0;
#         }
        
#         .footer {
#             position: fixed;
#             bottom: 0;
#             width: 100%;
#             background-color: #f0f2f6;
#             padding: 20px;
#             text-align: center;
#             border-top: 1px solid #ddd;
#         }
        
#         .footer-text {
#             margin: 0;
#         }
        
#         .footer-subtext {
#             margin: 5px 0 0 0;
#             font-size: 0.8em;
#             color: #666;
#         }
#         </style>
#     """, unsafe_allow_html=True)

# def main():
#     st.set_page_config(layout="wide", page_title="Bangladesh Vegetable Market Prediction")
#     set_custom_style()  # Apply custom styles
    
#     try:
#         # Load data directly from data.csv
#         data_path = "data.csv"
#         if not os.path.exists(data_path):
#             st.error("data.csv not found in the current directory!")
#             return
            
#         data = preprocess_data(data_path)
#         page = st.sidebar.selectbox("Select Page", ["Home", "Data Analysis"])
        
#         if page == "Home":
#             show_home_page(data)
#         else:
#             show_analysis_page(data)
            
#     except Exception as e:
#         st.error(f"Error loading data: {str(e)}")

# def show_home_page(data):
#     # Header
#     st.image("vegetable.jpg", caption="Fresh Vegetables Market", use_column_width=True)
#     st.title("Vegetable Price Prediction System")
    
#     # Input form
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         vegetable = st.selectbox(
#             "Select Vegetable",
#             ["Potato", "Onion", "Garlic", "Chili", "Ginger", "Egg"]
#         )
    
#     with col2:
#         prediction_date = st.date_input("Select Date")
    
#     with col3:
#         natural_disaster = st.checkbox("Natural Disaster")
#         political_issue = st.checkbox("Political Issue")
    
#     if st.button("Predict Price"):
#         # Train models and make predictions for both min and max prices
#         model_min, model_max, poly, train_score_min, test_score_min, train_score_max, test_score_max = create_model(vegetable, data)
#         predicted_min_price, predicted_max_price = predict_price(
#             model_min, model_max, poly, prediction_date, int(natural_disaster)
#         )
        
#         st.success(f"Predicted price range for {vegetable}: Min: ৳{predicted_min_price:.2f}, Max: ৳{predicted_max_price:.2f}")
        
#         col1, col2 = st.columns(2)
#         with col1:
#             st.metric("Model Accuracy (Training Min)", f"{train_score_min:.2%}")
#         with col2:
#             st.metric("Model Accuracy (Testing Min)", f"{test_score_min:.2%}")
#         with col1:
#             st.metric("Model Accuracy (Training Max)", f"{train_score_max:.2%}")
#         with col2:
#             st.metric("Model Accuracy (Testing Max)", f"{test_score_max:.2%}")
    
#     # Current prices cards
#     st.subheader("Current Market Prices")
#     latest_data = data.iloc[-1]
    
#     items = [
#         ("Potato", "🥔"), ("Onion", "🧅"), ("Garlic", "🧄"),
#         ("Chili", "🌶️"), ("Ginger", "🫑"), ("Egg", "🥚")
#     ]
    
#     cols = st.columns(3)
#     for idx, (item, emoji) in enumerate(items):
#         with cols[idx // 2]:
#             st.markdown(
#                 f"""
#                 <div class="veggie-card">
#                     <h3 class="veggie-title">{emoji} {item}</h3>
#                     <p class="price-text">Min: ৳{latest_data[f'{item}_min']}</p>
#                     <p class="price-text">Max: ৳{latest_data[f'{item}_max']}</p>
#                 </div>
#                 """,
#                 unsafe_allow_html=True
#             )
    
#     # Footer
#     st.markdown(
#         """
#         <div class="footer">
#             <p class="footer-text">© 2024 Bangladesh Vegetable Market Prediction System</p>
#             <p class="footer-subtext">Powered by Machine Learning</p>
#         </div>
#         """,
#         unsafe_allow_html=True
#     )


# def show_analysis_page(data):
#     st.title("Market Data Analysis")
    
#     vegetable = st.selectbox(
#         "Select Vegetable for Analysis",
#         ["Potato", "Onion", "Garlic", "Chili", "Ginger", "Egg"]
#     )
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         fig = px.line(
#             data,
#             x="DATE",
#             y=[f"{vegetable}_min", f"{vegetable}_max"],
#             title=f"{vegetable} Price Trends",
#             labels={"value": "Price (৳)", "variable": "Price Type"},
#             template="plotly_white"
#         )
#         fig.update_layout(
#             legend_title_text="Price Range",
#             hovermode="x unified"
#         )
#         st.plotly_chart(fig, use_container_width=True)
    
#     with col2:
#         model, poly, train_score, test_score = create_model(vegetable, data)
#         data[f'{vegetable}_predicted'] = model.predict(
#             poly.transform(data[['day_of_year', 'month', 'Natural_disaster']])
#         )
        
#         fig2 = px.line(
#             data,
#             x="DATE",
#             y=[f"{vegetable}_max", f"{vegetable}_predicted"],
#             title=f"{vegetable} Actual vs Predicted Prices",
#             labels={"value": "Price (৳)", "variable": "Price Type"},
#             template="plotly_white"
#         )
#         fig2.update_layout(
#             legend_title_text="Price Type",
#             hovermode="x unified"
#         )
#         st.plotly_chart(fig2, use_container_width=True)
    
#     st.subheader("Price Statistics")
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         avg_price = data[f"{vegetable}_max"].mean()
#         st.metric("Average Price", f"৳{avg_price:.2f}")
    
#     with col2:
#         max_price = data[f"{vegetable}_max"].max()
#         st.metric("Highest Price", f"৳{max_price:.2f}")
    
#     with col3:
#         min_price = data[f"{vegetable}_min"].min()
#         st.metric("Lowest Price", f"৳{min_price:.2f}")
    
#     st.subheader("Historical Data")
#     st.dataframe(
#         data[[
#             'DATE', f'{vegetable}_min', f'{vegetable}_max',
#             'Natural_disaster'
#         ]].style.highlight_max(axis=0, subset=[f'{vegetable}_max']),
#         use_container_width=True
#     )

# if __name__ == "__main__":
#     main()
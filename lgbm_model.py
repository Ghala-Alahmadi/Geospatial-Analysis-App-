# ----------------------
import streamlit as st
import joblib

import sklearn
import numpy as np
import pandas as pd
import pyproj
from sklearn.model_selection import train_test_split
import geopandas as gpd
from shapely.geometry import Point
from scipy.optimize import minimize
import contextily as ctx
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
import altair as alt
from datetime import datetime, timedelta

# ----------------------
def geometric_median(points: list):
    points = np.array(points)
    initial_guess = np.mean(points, axis=0)
    def objective_function(center):
        return np.sum(np.linalg.norm(points - center, axis=1))
    result = minimize(objective_function, initial_guess, method='Nelder-Mead')
    return result.x

def load_data(file_path):
    dataset = pd.read_csv(file_path)
    return dataset

def feature_engineering(dataset):
    dataset['OBSERVATION_DATE'] = pd.to_datetime(dataset['OBSERVATION_DATE'])
    dataset['DAY_OF_YEAR'] = dataset['OBSERVATION_DATE'].dt.dayofyear
    dataset['HOUR'] = dataset['OBSERVATION_DATE'].dt.hour
    coords = list(zip(dataset['LATITUDE'], dataset['LONGITUDE']))
    center = geometric_median(coords)
    dataset['dist_bearing'] = dataset.apply(lambda row: get_dist_bearing(center, (row['LATITUDE'], row['LONGITUDE'])), axis=1)
    dataset['DIST'], dataset['DIREC'] = zip(*dataset['dist_bearing'])
    return dataset

def data_cleaning(dataset):
    dataset = dataset.drop_duplicates()
    return dataset

def get_dist_bearing(center, point):
    geodisc = pyproj.Geod(ellps='WGS84')
    lon1, lat1 = center
    lon2, lat2 = point
    fwd_azimuth, back_azimuth, distance = geodisc.inv(lon1, lat1, lon2, lat2)
    return distance, fwd_azimuth

def main():
    st.title("Geospatial Analysis App!")
    st.sidebar.title("Menu")
    menu = st.sidebar.selectbox("Select an option", ["Overview", "Data Analysis", "Temperature Prediction"])

    if menu == "Overview":
        st.subheader("Overview")
        st.write("Welcome to the Geospatial Analysis App!")
        st.write("Geospatial analysis is a powerful approach for examining and interpreting data that possesses a spatial or geographical component. By leveraging geographic information systems (GIS) and various analytical techniques, geospatial analysis allows us to uncover valuable insights about our world.")
        st.write("In this project, our focus is on exploring climate data within the context of the Kingdom of Saudi Arabia. By harnessing geospatial analysis techniques, we aim to gain a deeper understanding of the intricate relationships between climatic factors and geographical features within this region.")
        st.write("The Kingdom of Saudi Arabia, with its vast and diverse landscape, presents an intriguing dataset for geospatial analysis. Across different geographical locations offers a rich tapestry for exploration.")
        st.write("Through this app, we'll delve into climate data sourced from various observation stations across Saudi Arabia. By examining parameters such as temperature, elevation, and humidity, we'll unravel spatial patterns and correlations, shedding light on the complex interplay between environmental factors.")
        st.write("Join us on this journey as we employ geospatial analysis techniques to decipher the climate data of the Kingdom of Saudi Arabia, uncovering insights that may help scientific research.")

    elif menu == "Data Analysis":
        st.subheader("Data Analysis Section")
        st.write("Welcome to the Data Analysis section! In this section, we will explore various geospatial analysis techniques to gain insights into climate data related to the Kingdom of Saudi Arabia.")
        st.write("Geospatial analysis involves analyzing and visualizing data that has a geographical or spatial component. It helps us understand how different climatic parameters vary across space and time.")
        st.write("In this project, we are particularly interested in studying climate data, including temperature, elevation, and humidity, to uncover spatial patterns and correlations in the Kingdom of Saudi Arabia.")
        st.write("One of the key techniques we'll be using is kriging, a geostatistical method for interpolating spatial data. However, since kriging is not directly supported by Streamlit's mapping functionality, we'll focus on other regression models for analysis.")

        file_path = 'modified_dataset.csv'
        dataset = load_data(file_path)
        dataset = feature_engineering(dataset)
        dataset = data_cleaning(dataset)

        features = [
            'DIST',
            'DIREC',
            'ELEVATION',
            'DAY_OF_YEAR',
            'HOUR',
            'AIR_TEMPERATURE_DEW_POINT']

        st.write("Let's start by exploring regression models for predicting air temperature based on various features.")
        st.write("We'll evaluate the following regression models:")
        st.write("And after training and testing each model, we'll compare their performance using R-squared (R2).")
        models = {
            'KNN': KNeighborsRegressor(),
            'LightGBM': lgb.LGBMRegressor(),
            'Ridge Regression': Ridge(),
            'Gradient Boosting Regressor': GradientBoostingRegressor()}

        st.write("Here are our models ranked based on the result after training:")
        for model_name in models:
            st.write(f"- {model_name}")
            

        st.write("In conclusion, while KNN and LightGBM emerged as the top-performing models, and they are the most appropriate models for predicting air temperature based on the given features")

    elif menu == "Temperature Prediction":
        st.subheader("Temperature Prediction with Categorization")

        #Load the LightGBM model, the reason that we choose LightGBM because after tuning, there is improvement in R-squared on the training set,and it's compatible with both small and large datasets.
        model = joblib.load('lgbm_model.pkl')

        st.write("Enter numeric values to predict the temperature:")
        
        # User input - input_values
        start_date = datetime.today() - timedelta(days=365*24)
        end_date = datetime.today()
        
        Longitude_values = st.text_input('Enter Longitude:', value='0.0')
        Latitude_values = st.text_input('Enter Latitude:', value='0.0')
        Elevation_values = st.text_input('Enter Elevation(Integer):', value='0')
        Day_Of_Year_values = st.date_input("Choose the Day of the Year:",value=end_date, min_value=start_date, max_value=end_date)
        Hour_values = st.selectbox("Choose the Hour :", [str(hour) for hour in range(1, 25)])
        day_of_year = Day_Of_Year_values.timetuple().tm_yday
        
        # Categorization function
        def categorize_temperature(temp):
            if temp < 12:
                return "Cold"
            elif 12 <= temp < 25:
                return "Normal"
            else:
                return "Hot"

        # Prediction + categorization
        if st.button('Predict Temperature'):
          
            try:
                all_inputs = np.array([[float(Longitude_values), float(Latitude_values), int(Elevation_values), day_of_year, float(Hour_values)]], dtype=np.float32)                #input_list = np.array([[float(x) for x in input_values.split(',')]])
                prediction = model.predict(all_inputs)[0]
                category = categorize_temperature(prediction)
                st.write(f"Predicted Air Temperature (LightGBM): {prediction:.2f} °C")
                st.write(f"Temperature Category: {category}")
            except ValueError:
                st.write("Invalid input. Please enter numeric values")

if __name__ == "__main__":
    main()

# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# import folium
# from streamlit_folium import st_folium

# # Load and prepare dataset
# df = pd.read_csv("Indian_earthquake_data.csv")

# # Convert relevant columns to numeric
# cols_to_convert = ['Estimated Economic Loss (USD)', 'Population Density', 
#                    'Total Fatalities', 'Buildings Collapsed', 'Magnitude', 'Depth']
# for col in cols_to_convert:
#     df[col] = pd.to_numeric(df[col], errors='coerce')

# # Drop invalid rows
# df.dropna(subset=cols_to_convert + ['Latitude', 'Longitude'], inplace=True)

# # Filter valid data
# df = df[(df['Estimated Economic Loss (USD)'] >= 0) & 
#         (df['Population Density'] > 0) &
#         (df['Total Fatalities'] >= 0) & 
#         (df['Buildings Collapsed'] >= 0) & 
#         (df['Magnitude'] > 0) & 
#         (df['Depth'] >= 0)]

# # Estimate affected population
# df['Estimated Population'] = df['Population Density'] * 1.5

# # Features and target
# features = ['Magnitude', 'Depth', 'Population Density', 'Buildings Collapsed']
# X = df[features]
# y = df['Total Fatalities']

# # Train model
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model.fit(X, y)

# # Prediction function
# def predict_casualties(magnitude, depth, population_density, buildings_collapsed):
#     sample = pd.DataFrame([[magnitude, depth, population_density, buildings_collapsed]], columns=features)
#     prediction = model.predict(sample)[0]
#     estimated_population = population_density * 1.5
#     return int(min(prediction, estimated_population))

# # Map plotting function
# def plot_earthquake_map(df):
#     m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=5)
#     for _, row in df.iterrows():
#         mag = row['Magnitude']
#         popup_info = f"""
#         <b>Location:</b> {row.get('Location', 'Unknown')}<br>
#         <b>Magnitude:</b> {mag}<br>
#         <b>Depth:</b> {row['Depth']} km<br>
#         <b>Fatalities:</b> {row['Total Fatalities']}<br>
#         <b>Buildings Collapsed:</b> {row['Buildings Collapsed']}
#         """
#         color = "green" if mag < 4 else "orange" if mag < 6 else "red"
#         folium.CircleMarker(
#             location=[row['Latitude'], row['Longitude']],
#             radius=6,
#             color=color,
#             fill=True,
#             fill_opacity=0.7,
#             popup=folium.Popup(popup_info, max_width=250)
#         ).add_to(m)
#     return m

# # Streamlit UI
# def main():
#     st.set_page_config(page_title="Earthquake Casualty Predictor", layout="centered")
#     st.title("🧠 Earthquake Casualty Predictor")
#     st.write("Estimate possible casualties based on earthquake parameters and visualize past events on a map.")

#     # Prediction Input
#     st.subheader("📊 Enter Earthquake Details")
#     magnitude = st.number_input("Earthquake Magnitude", min_value=0.0, step=0.1)
#     depth = st.number_input("Depth (km)", min_value=0.0, step=1.0)
#     population_density = st.number_input("Population Density (per sq km)", min_value=0.0, step=1.0)
#     buildings_collapsed = st.number_input("Estimated Buildings Collapsed", min_value=0, step=1)

#     # Add input fields for latitude and longitude
#     latitude = st.number_input("Latitude of Earthquake (°)", min_value=-90.0, max_value=90.0, step=0.1)
#     longitude = st.number_input("Longitude of Earthquake (°)", min_value=-180.0, max_value=180.0, step=0.1)

#     if st.button("Predict Casualties"):
#         if magnitude and depth and population_density:
#             casualties = predict_casualties(magnitude, depth, population_density, buildings_collapsed)
#             st.success(f"🔮 Predicted Casualties: **{casualties} people**")
#         else:
#             st.error("⚠️ Please fill all input fields.")

#     # Map Output
#     st.markdown("---")
#     st.subheader("🗺️ Earthquake Map: Visualize Impact Zones")
#     st.write("Each marker represents a recorded earthquake in the dataset. Color-coded by severity.")

#     # Plot the user-provided location on the map
#     if latitude and longitude:
#         user_map = folium.Map(location=[latitude, longitude], zoom_start=5)
#         folium.Marker([latitude, longitude], popup="User Input Location").add_to(user_map)
#         st_folium(user_map, width=700, height=500)

#     # Map of past earthquakes (existing dataset)
#     earthquake_map = plot_earthquake_map(df)
#     st_folium(earthquake_map, width=700, height=500)

# if __name__ == "__main__":
#     main()









import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import folium
from streamlit_folium import st_folium

# Set Streamlit page configuration
st.set_page_config(page_title="Earthquake Casualty Predictor", layout="centered")

# Load and prepare dataset
df = pd.read_csv("Indian_earthquake_data.csv")

# Convert relevant columns to numeric
cols_to_convert = ['Estimated Economic Loss (USD)', 'Population Density', 
                   'Total Fatalities', 'Buildings Collapsed', 'Magnitude', 'Depth']
for col in cols_to_convert:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop invalid rows
df.dropna(subset=cols_to_convert + ['Latitude', 'Longitude'], inplace=True)

# Filter valid data
df = df[(df['Estimated Economic Loss (USD)'] >= 0) & 
        (df['Population Density'] > 0) & 
        (df['Total Fatalities'] >= 0) & 
        (df['Buildings Collapsed'] >= 0) & 
        (df['Magnitude'] > 0) & 
        (df['Depth'] >= 0)]

# Estimate affected population
df['Estimated Population'] = df['Population Density'] * 1.5

# Features and target
features = ['Magnitude', 'Depth', 'Population Density', 'Buildings Collapsed']
X = df[features]
y = df['Total Fatalities']

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Prediction function
def predict_casualties(magnitude, depth, population_density, buildings_collapsed):
    sample = pd.DataFrame([[magnitude, depth, population_density, buildings_collapsed]], columns=features)
    prediction = model.predict(sample)[0]
    estimated_population = population_density * 1.5
    return int(min(prediction, estimated_population))

# Map plotting function
def plot_earthquake_map(df):
    m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=5)
    for _, row in df.iterrows():
        mag = row['Magnitude']
        popup_info = f"""
        <b>Location:</b> {row.get('Location', 'Unknown')}<br>
        <b>Magnitude:</b> {mag}<br>
        <b>Depth:</b> {row['Depth']} km<br>
        <b>Fatalities:</b> {row['Total Fatalities']}<br>
        <b>Buildings Collapsed:</b> {row['Buildings Collapsed']}
        """
        color = "green" if mag < 4 else "orange" if mag < 6 else "red"
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=6,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(popup_info, max_width=250)
        ).add_to(m)
    return m

# Streamlit UI
def main():
    st.title("🧠 Earthquake Casualty Predictor")
    st.write("Estimate possible casualties based on earthquake parameters and visualize past events on a map.")

    # Prediction Input
    st.subheader("📊 Enter Earthquake Details")
    magnitude = st.number_input("Earthquake Magnitude", min_value=0.0, step=0.1)
    depth = st.number_input("Depth (km)", min_value=0.0, step=1.0)
    population_density = st.number_input("Population Density (per sq km)", min_value=0.0, step=1.0)
    buildings_collapsed = st.number_input("Estimated Buildings Collapsed", min_value=0, step=1)

    # Add input fields for latitude and longitude
    latitude = st.number_input("Latitude of Earthquake (°)", min_value=-90.0, max_value=90.0, step=0.1)
    longitude = st.number_input("Longitude of Earthquake (°)", min_value=-180.0, max_value=180.0, step=0.1)

    if st.button("Predict Casualties"):
        if magnitude and depth and population_density:
            casualties = predict_casualties(magnitude, depth, population_density, buildings_collapsed)
            st.success(f"🔮 Predicted Casualties: **{casualties} people**")
        else:
            st.error("⚠️ Please fill all input fields.")

    # Map Output
    st.markdown("---")
    st.subheader("🗺️ Earthquake Map: Visualize Impact Zones")
    st.write("Each marker represents a recorded earthquake in the dataset. Color-coded by severity.")

    # Plot the user-provided location on the map
    if latitude and longitude:
        user_map = folium.Map(location=[latitude, longitude], zoom_start=5)
        folium.Marker([latitude, longitude], popup="User Input Location").add_to(user_map)
        st_folium(user_map, width=700, height=500)

    # Map of past earthquakes (existing dataset)
    earthquake_map = plot_earthquake_map(df)
    st_folium(earthquake_map, width=700, height=500)

if __name__ == "__main__":
    main()

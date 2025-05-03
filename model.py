# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, r2_score
# import numpy as np

# # Load data
# df = pd.read_csv("Indian_earthquake_data.csv")

# # Drop rows with negative or missing values in important columns
# df = df[(df['Estimated Economic Loss (USD)'] >= 0) & 
#         (df['Population Density'] > 0) &
#         (df['Total Fatalities'] >= 0) &
#         (df['Buildings Collapsed'] >= 0) &
#         (df['Magnitude'] > 0) &
#         (df['Depth'] >= 0)]

# # Assume we have an estimated affected population (e.g., from Population Density)
# df['Estimated Population'] = df['Population Density'] * 1.5  # just a scaling assumption

# # Features and target
# features = ['Magnitude', 'Depth', 'Population Density', 'Buildings Collapsed']
# target = 'Total Fatalities'

# X = df[features]
# y = df[target]

# # Split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Model
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Predictions
# y_pred = model.predict(X_test)

# # Clamp predictions to max estimated population
# estimated_population_test = X_test['Population Density'] * 1.5
# y_pred_clamped = np.minimum(y_pred, estimated_population_test)

# # Evaluation
# print("🔮 Casualties Prediction (Post-corrected):")
# print("R² Score:", r2_score(y_test, y_pred_clamped))
# print("MSE:", mean_squared_error(y_test, y_pred_clamped))

# # Example prediction
# def predict_casualties(magnitude, depth, population_density, buildings_collapsed):
#     sample = pd.DataFrame([[magnitude, depth, population_density, buildings_collapsed]], columns=features)
#     pred = model.predict(sample)[0]
#     estimated_population = population_density * 1.5
#     return min(pred, estimated_population)

# # Try with example values
# example_prediction = predict_casualties(6.5, 20, 1500, 200)
# print(f"\n📌 Predicted Casualties (example): {int(example_prediction)} people")



import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix
import folium
from streamlit_folium import st_folium
import seaborn as sns
import matplotlib.pyplot as plt

# Load and prepare dataset
df = pd.read_csv("Indian_earthquake_data.csv")
cols_to_convert = ['Estimated Economic Loss (USD)', 'Population Density', 
                   'Total Fatalities', 'Buildings Collapsed', 'Magnitude', 'Depth']
for col in cols_to_convert:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df.dropna(subset=cols_to_convert + ['Latitude', 'Longitude'], inplace=True)
df = df[(df['Estimated Economic Loss (USD)'] >= 0) & 
        (df['Population Density'] > 0) &
        (df['Total Fatalities'] >= 0) & 
        (df['Buildings Collapsed'] >= 0) & 
        (df['Magnitude'] > 0) & 
        (df['Depth'] >= 0)]

df['Estimated Population'] = df['Population Density'] * 1.5

# Features and target
features = ['Magnitude', 'Depth', 'Population Density', 'Buildings Collapsed']
X = df[features]
y = df['Total Fatalities']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation on test set
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Binned confusion matrix
def bin_fatalities(y):
    return pd.cut(y, bins=[-1, 10, 50, 200, np.inf], labels=["Low", "Moderate", "High", "Severe"])

y_test_binned = bin_fatalities(y_test)
y_pred_binned = bin_fatalities(y_pred)
conf_mat = confusion_matrix(y_test_binned, y_pred_binned, labels=["Low", "Moderate", "High", "Severe"])

# Prediction function
def predict_casualties(magnitude, depth, population_density, buildings_collapsed):
    sample = pd.DataFrame([[magnitude, depth, population_density, buildings_collapsed]], columns=features)
    prediction = model.predict(sample)[0]
    estimated_population = population_density * 1.5
    return int(min(prediction, estimated_population))

# Map plotting
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
    st.set_page_config(page_title="Earthquake Casualty Predictor", layout="centered")
    st.title("🧠 Earthquake Casualty Predictor")
    st.write("Estimate possible casualties based on earthquake parameters and visualize past events on a map.")

    # Prediction Inputs
    st.subheader("📊 Enter Earthquake Details")
    magnitude = st.number_input("Earthquake Magnitude", min_value=0.0, step=0.1)
    depth = st.number_input("Depth (km)", min_value=0.0, step=1.0)
    population_density = st.number_input("Population Density (per sq km)", min_value=0.0, step=1.0)
    buildings_collapsed = st.number_input("Estimated Buildings Collapsed", min_value=0, step=1)

    latitude = st.number_input("Latitude of Earthquake (°)", min_value=-90.0, max_value=90.0, step=0.1)
    longitude = st.number_input("Longitude of Earthquake (°)", min_value=-180.0, max_value=180.0, step=0.1)

    if st.button("Predict Casualties"):
        if magnitude and depth and population_density:
            casualties = predict_casualties(magnitude, depth, population_density, buildings_collapsed)
            st.success(f"🔮 Predicted Casualties: *{casualties} people*")
        else:
            st.error("⚠ Please fill all input fields.")

    st.markdown("---")
    st.subheader("🗺 Earthquake Map: Visualize Impact Zones")
    st.write("Each marker represents a recorded earthquake in the dataset. Color-coded by severity.")
    if latitude and longitude:
        user_map = folium.Map(location=[latitude, longitude], zoom_start=5)
        folium.Marker([latitude, longitude], popup="User Input Location").add_to(user_map)
        st_folium(user_map, width=700, height=500)

    earthquake_map = plot_earthquake_map(df)
    st_folium(earthquake_map, width=700, height=500)

    # Evaluation Metrics Display
    st.markdown("---")
    st.subheader("📈 Model Performance Summary")
    st.write(f"*MAE*: {mae:.2f} casualties")
    st.write(f"*RMSE*: {rmse:.2f} casualties")
    st.write(f"*R² Score*: {r2:.2f}")

    st.subheader("🔢 Confusion Matrix (Binned Categories)")
    fig, ax = plt.subplots()
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Low", "Moderate", "High", "Severe"],
                yticklabels=["Low", "Moderate", "High", "Severe"],
                ax=ax)
    ax.set_xlabel("Predicted Category")
    ax.set_ylabel("Actual Category")
    st.pyplot(fig)

if _name_ == "_main_":
    main()
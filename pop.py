import pandas as pd

# Load your dataset
df = pd.read_csv("Indian_earthquake_data.csv")

# Coordinates to filter
target_lat = 29.0600
target_lon = 77.4200

# Filter rows with the same coordinates
filtered_df = df[(df["Latitude"] == target_lat) & (df["Longitude"] == target_lon)]

# Calculate total population
total_population = filtered_df["Population Density"].sum()

print(f"Total population at ({target_lat}, {target_lon}): {total_population}")

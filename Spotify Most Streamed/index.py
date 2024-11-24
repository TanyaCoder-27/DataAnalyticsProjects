import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the dataset
file_path = "universal_top_spotify_songs.csv"  # Update this path
data = pd.read_csv(file_path)

# Display dataset overview
print("Dataset Head:")
print(data.head())

print("\nDataset Info:")
print(data.info())

# Preprocess the data
print("\nChecking for Missing Values:")
missing_values = data.isnull().sum()
print(missing_values)

# Handle missing values (example: drop rows with NA)
data_cleaned = data.dropna()

# Basic data analysis
top_songs = data_cleaned.groupby("name")["popularity"].sum().sort_values(ascending=False).head(10)
print("\nTop 10 Songs by Popularity:")
print(top_songs)

# Visualization: Top 10 Songs by Popularity
plt.figure(figsize=(10, 6))
top_songs.plot(kind="bar", color="skyblue")
plt.title("Top 10 Songs by Popularity")
plt.xlabel("Song Name")
plt.ylabel("Total Popularity")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# The following Python codes generate the dataset called 'road_network.csv' which has 200 cities and 3,000 unique connections.

import random
import pandas as pd
from faker import Faker

# Initialize Faker
fake = Faker()

# Read city names from the text file
with open('city_names_list.txt', 'r') as file:
    cities = file.read().splitlines()

# Ensure we have at least 200 unique cities
if len(cities) < 200:
    # Generate additional cities using Faker to make up for missing cities
    additional_cities_needed = 200 - len(cities)
    generated_cities = [fake.city() for _ in range(additional_cities_needed)]
    cities.extend(generated_cities)

# Limit to exactly 200 unique cities
cities = list(set(cities))[:200]  # Convert to a set and back to list to ensure uniqueness, then slice to 200

num_cities = 200
num_raw_rows = 10000
num_final_edges = 3000

# Function to generate unique city pairs
def generate_city_pairs(cities, num_pairs):
    pairs = set()
    while len(pairs) < num_pairs:
        city1, city2 = random.sample(cities, 2)
        if city1 != city2:
            pairs.add(tuple(sorted((city1, city2))))
    return list(pairs)

# Generate 10,000 raw city pairs with distances
raw_data = []
for _ in range(num_raw_rows):
    city1, city2 = random.sample(cities, 2)
    distance = fake.random_int(min=50, max=500)  # Random distance between 50 and 500 miles
    raw_data.append((city1, city2, distance))

# Create DataFrame
df = pd.DataFrame(raw_data, columns=['city1', 'city2', 'distance'])

# Remove duplicates and self-connections
df['sorted_pair'] = df.apply(lambda row: tuple(sorted([row['city1'], row['city2']])), axis=1)
df = df.drop_duplicates(subset=['sorted_pair']).drop(columns=['sorted_pair'])
df = df[df['city1'] != df['city2']]

# Ensure 3,000 unique connections
if len(df) < num_final_edges:
    raise ValueError("Not enough unique connections. Increase the number of raw rows or adjust criteria.")

df = df.head(num_final_edges)

# Verify the dataset
num_unique_cities = max(df['city1'].nunique(), df['city2'].nunique())
num_unique_edges = len(df)
print(f"Number of unique cities: {num_unique_cities}")
print(f"Number of unique edges: {num_unique_edges}")

assert num_unique_cities == num_cities, f"Number of unique cities is not {num_cities}. Found {num_unique_cities} unique cities."
assert num_unique_edges == num_final_edges, f"Number of unique edges is not {num_final_edges}. Found {num_unique_edges} unique edges."

# Save the dataset to CSV
df.to_csv('road_network.csv', index=False)
print("Dataset generated and saved as 'road_network.csv'")

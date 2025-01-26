# First, we create a database named "citynetwork" in the DBMS named 'City-network' in Neo4j Desktop
# Then we run this python file to upload the data to the database
# Lastly, we run the Neo4j Shell Scripts (Cypher Queries) in Neo4j Desktop and take screenshots of the outputs
# Brief explanations of the queries and their ouput are given in the document file named 'Results_Documentation'.

import pandas as pd
from neo4j import GraphDatabase

# Neo4j connection details
uri = "bolt://localhost:7687"  # Update with your Neo4j URI
username = "neo4j"             # Update with your Neo4j username
password = "12345678"          # Update with your Neo4j password
database_name = "citynetwork"  # Specify the database name

# Create a driver instance
driver = GraphDatabase.driver(uri, auth=(username, password))

# Initialize dataset
# Function to upload to Neo4j in batches of 1000
def upload_to_neo4j_in_batches(file_path, batch_size=1000):
    # Read the CSV file in chunks
    for chunk in pd.read_csv(file_path, chunksize=batch_size):
        with driver.session(database=database_name) as session:  # Use specific database
            # Convert the chunk into Cypher statements
            for index, row in chunk.iterrows():
                # Escape single quotes in city names to avoid Cypher syntax issues
                city1 = row['city1'].replace("'", "\\'")
                city2 = row['city2'].replace("'", "\\'")
                
                cypher_query = f"""
                MERGE (c1:City {{name: '{city1}'}})
                MERGE (c2:City {{name: '{city2}'}})
                MERGE (c1)-[r:ROAD {{distance: {int(row['distance'])}}}]->(c2)
                """
                try:
                    session.run(cypher_query)
                except Exception as e:
                    print(f"Error uploading row {index}: {e}")
                    continue
        print(f"Uploaded batch with {len(chunk)} rows.")

# Path to the filtered CSV file
filtered_csv_path = 'road_network.csv'

# Upload the filtered dataset to Neo4j in batches
upload_to_neo4j_in_batches(filtered_csv_path)
print("Filtered dataset uploaded to Neo4j in batches.")



print("\n\n#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Task 1 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

# GOAL: List Roads from/to a Specific City (for example: Atlanta) with Distances and Destinations

# Neo4j Shell Script:
'''
MATCH (c:City)-[r:ROAD]->(d:City)
WHERE c.name='Atlanta'
RETURN c.name AS From_City, d.name AS To_City, r.distance AS Distance
UNION
MATCH (d:City)-[r:ROAD]->(c:City)
WHERE c.name='Atlanta'
RETURN d.name AS From_City, c.name AS To_City, r.distance AS Distance
'''
# Functionality: This query searches for all nodes labeled City that are connected to the specified city "Atlanta" via a ROAD relationship.
# It returns the names of the destination cities and the distances of these roads, distinguishing whether the roads start from or end at "Atlanta".



print("\n\n#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Task 2 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

# GOAL: Find Roads Longer than a Specific Distance

# Neo4j Shell Script:
'''
MATCH (c1:City)-[r:ROAD]->(c2:City)
WHERE r.distance > 100
RETURN c1.name AS City1, c2.name AS City2, r.distance AS Distance
'''
# Functionality: The query looks for ROAD relationships between nodes labeled City where the distance property of the relationship exceeds 100.
# It then returns the names of the connected cities and the distance of each road that meets the criteria.



print("\n\n#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Task 3 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

# GOAL: Count the Number of Roads Connected to a Specific City (for example: Atlanta)

# Neo4j Shell Script:
'''
MATCH (c:City {name: 'Atlanta'})-[:ROAD]-()
RETURN c.name AS City, COUNT(*) AS Number_of_Connections
'''
# Functionality: This query finds all nodes labeled City that have any ROAD relationships to or from the specified city "Atlanta".
# It returns the name of the city and the count of all its connected roads. 



print("\n\n#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Task 4 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

# GOAL: Find Cities Connected to a City with a Distance Below a Specific Value

# Neo4j Shell Script:
'''
MATCH (c1:City)-[r:ROAD]->(c2:City)
WHERE c1.name = 'New York' AND r.distance < 200
RETURN c1.name AS From_City, c2.name AS Connected, r.distance AS Distance
UNION
MATCH (c2:City)-[r:ROAD]->(c1:City)
WHERE c1.name = 'New York' AND r.distance < 200
RETURN c2.name AS From_City, c1.name AS Connected, r.distance AS Distance
'''
# Functionality: This query returns all of the cities that are connected to New York City within 200 km.



print("\n\n#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Task 5 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

# GOAL: List All Roads Leading to a Specific City

# Neo4j Shell Script:
'''
MATCH (c1:City)-[r:ROAD]->(c2:City)
WHERE c2.name = 'Detroit'
RETURN c1.name AS From_City, c2.name AS To_City, r.distance AS Distance
'''
# Functionality: This query return all of the roads that lead to Detroit.



print("\n\n#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Task 6 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

# GOAL: Calculate the Total Distance of All Roads Connected to a City

# Neo4j Shell Script:
'''
MATCH (c1:City)-[r:ROAD]->(c2:City)
WHERE c1.name = 'Detroit' OR c2.name = 'Detroit'
RETURN 'Detroit' AS City, SUM(r.distance) AS TotalDistance
'''
# Functionality: This query return the total distance of all of the roads that lead to Detroit.



print("\n\n#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Task 7 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

# GOAL: Find the Shortest Road Between Two Cities

# Neo4j Shell Script:
'''
MATCH (c1:City)-[r:ROAD]->(c2:City)
WHERE c1.name = 'Phoenix' AND c2.name = 'Buffalo'
RETURN c1.name AS From_City, c2.name AS To_City, r.distance AS Distance
ORDER BY r.distance ASC
Limit 1
'''
# Functionality: This query shows the shortest road between the two cities Buffalo and Pheonix.



print("\n\n#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Task 8 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

# GOAL: Find Cities Connected to Both 'Denver' and 'Seattle'

# Neo4j Shell Script:
'''
MATCH (city:City)-[:DISTANCE]->(connected:City)
WHERE city.name IN ['Denver', 'Seattle']
RETURN DISTINCT connected.name AS ConnectedCity;
'''
# Functionality: This query finds cities directly connected to both "Denver" and "Seattle".
# Thus it identifies mutual connection points that link the two cities in the road network.



print("\n\n#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Task 9 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

# GOAL: List Cities with More Than three Connections

# Neo4j Shell Script:
'''
MATCH (city:City)-[r:DISTANCE]->()
WITH city, COUNT(r) AS connections
WHERE connections > 3
RETURN city.name AS City, connections;
'''
# Functionality: This query lists cities with more than three direct connections,
# thus highlighting key hubs with significant connectivity.



print("\n\n#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Task 10 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

# GOAL: Calculate the total distance of the network

# Neo4j Shell Script:
'''
MATCH ()-[r:DISTANCE]->()
RETURN SUM(r.miles) AS TotalNetworkDistance;
'''
# Functionality: This query calculates the total distance of all roads in the network
# and provides a measure of the network's overall scale.



print("\n\n#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Extra credit tasks  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

print("\n\n#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Task E1 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

# GOAL: Identify Cities with Exactly Two Connections

# Neo4j Shell Script:
'''
MATCH (p:Node)-[r]->()
WITH p.name AS Node, count(r) AS RelationshipCount
WHERE RelationshipCount = 2
RETURN Node AS city_with_two_connections;
'''

# For getting no output, we change the count value
'''
MATCH (p:Node)-[r]->()
WITH p.name AS Node, count(r) AS RelationshipCount
WHERE RelationshipCount = 7
RETURN Node AS city_with_two_connections;
'''
# Functionality: This query finds all cities that have exactly two outgoing connections. 
# It uses the COUNT function to count the number of outgoing relationships per city and returns the city names that meet the criteria.



print("\n\n#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Task E2 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

# GOAL: List Cities with Both Incoming and Outgoing Connections

# Neo4j Shell Script:
'''
MATCH (p)-[r1]->()
WITH p, count(r1) AS outgoing_count
MATCH (p)<-[r2]-()
WITH p, outgoing_count, count(r2) AS incoming_count
WHERE outgoing_count > 0 AND incoming_count > 0
RETURN p.name AS Node, incoming_count, outgoing_count
LIMIT 10;
'''
# Functionality: This query finds all cities that have at least one incoming and one outgoing connection. 
# It first counts outgoing connections and then counts incoming connections. Only cities with both incoming and outgoing connections are returned along with their respective counts.



print("\n\n#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Task E3 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

# GOAL: Identify the Top 3 Pairs of Cities with the Longest Direct Distances

# Neo4j Shell Script:
'''
MATCH (c1:Node)-[r]->(c2:Node)
RETURN c1.name AS city1, c2.name AS city2, r.type
ORDER BY r.type DESC
LIMIT 3;
'''
# Functionality: This query finds the top three pairs of cities with the longest direct distances. 
# It sorts the relationships by the distance property in descending order and limits the result to the top three entries, returning both city names and the distance.

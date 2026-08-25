from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

# Load .env and use environment variables for connection
load_dotenv()

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD")

if PASSWORD is None:
    raise EnvironmentError("NEO4J_PASSWORD is not set. Please create a .env file or set environment variables.")

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

# 2. The Cypher query to insert the new predicted links
insert_query = """
UNWIND $predictions AS row
MATCH (p1:Person {name: row.actor1})
MATCH (p2:Person {name: row.actor2})
MERGE (p1)-[r:PREDICTED_CO_ACTOR]->(p2)
SET r.confidence = row.probability
"""

import pandas as pd

# Mocking the dataframe with dummy data for testing purposes
predicted_links_df = pd.DataFrame([
    {'u': 'Tom Hanks', 'v': 'Meg Ryan', 'prediction': 1, 'predict_proba': 0.95},
    {'u': 'Keanu Reeves', 'v': 'Carrie-Anne Moss', 'prediction': 1, 'predict_proba': 0.88},
    {'u': 'Brad Pitt', 'v': 'Edward Norton', 'prediction': 0, 'predict_proba': 0.20}
])

# 3. Filter your dataframe to ONLY include the links the model predicted as '1' (True)
positive_predictions = predicted_links_df[predicted_links_df['prediction'] == 1]

# 4. Convert dataframe to a list of dictionaries for Neo4j
records_to_insert = positive_predictions.rename(
    columns={'u': 'actor1', 'v': 'actor2', 'predict_proba': 'probability'}
).to_dict('records')

# 5. Execute the write operation
with driver.session() as session:
    session.run(insert_query, predictions=records_to_insert)
    print(f"Successfully wrote {len(records_to_insert)} predicted links to Neo4j!")
    
driver.close()
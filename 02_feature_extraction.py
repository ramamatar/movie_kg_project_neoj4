import os
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load .env
load_dotenv()

# Use environment variables for Neo4j connection
URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD")

if PASSWORD is None:
    raise EnvironmentError("NEO4J_PASSWORD is not set. Please create a .env file or set environment variables.")

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

def build_coactor_network():
    print("Extracting Person-Movie graph to project co-actor network...")
    # Fetch top 1000 actors and their movies to avoid memory limits locally
    query = """
    MATCH (p1:Person)-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(p2:Person)
    WHERE id(p1) < id(p2)
    RETURN p1.name AS actor1, p2.name AS actor2, count(m) AS weight
    """
    
    G = nx.Graph()
    with driver.session() as session:
        result = session.run(query)
        for record in result:
            G.add_edge(record["actor1"], record["actor2"], weight=record["weight"])
            
    print(f"Graph loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

def extract_topological_features(G):
    print("Calculating topological features...")
    
    # 1. Degree Centrality
    degree_cent = nx.degree_centrality(G)
    
    # 2. PageRank
    pagerank = nx.pagerank(G, weight='weight')
    
    # 3. Betweenness Centrality (sample nodes for speed if graph is large)
    k = min(500, G.number_of_nodes())
    betweenness = nx.betweenness_centrality(G, weight='weight', k=k)
    
    # 4. Community Detection (Louvain)
    # Using NetworkX built-in louvain
    communities = nx.community.louvain_communities(G, weight='weight')
    
    # Map community ID to nodes
    comm_map = {}
    for i, comm in enumerate(communities):
        for node in comm:
            comm_map[node] = i
            
    # Combine into DataFrame
    features = []
    for node in G.nodes():
        features.append({
            "node": node,
            "degree_centrality": degree_cent.get(node, 0),
            "pagerank": pagerank.get(node, 0),
            "betweenness_centrality": betweenness.get(node, 0),
            "community_id": comm_map.get(node, -1)
        })
        
    df = pd.DataFrame(features)
    df.to_csv("node_features.csv", index=False)
    print("Saved topological features to node_features.csv")
    return df, G

def visualize_degree_distribution(G):
    degrees = [d for n, d in G.degree()]
    plt.figure(figsize=(8, 6))
    plt.hist(degrees, bins=50, color='coral', edgecolor='black')
    plt.title("Degree Distribution of Co-Actor Network")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.yscale('log')
    plt.savefig("degree_distribution.png", dpi=300)
    print("Saved degree_distribution.png")
    plt.close()

if __name__ == "__main__":
    G = build_coactor_network()
    df, G = extract_topological_features(G)
    visualize_degree_distribution(G)
    driver.close()

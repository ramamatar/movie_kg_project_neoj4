import os
import matplotlib.pyplot as plt
import networkx as nx
import neo4j
from dotenv import load_dotenv

# Load environment variables from .env (if present)
load_dotenv()

# Configure connection (do NOT hardcode credentials here)
# Set `NEO4J_URI`, `NEO4J_USER`, and `NEO4J_PASSWORD` in a local .env file or environment
URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD")

if PASSWORD is None:
    raise EnvironmentError("NEO4J_PASSWORD is not set. Please create a .env file or set environment variables.")

driver = neo4j.GraphDatabase.driver(URI, auth=(USER, PASSWORD))

def get_schema_info():
    print("Fetching schema information...")
    with driver.session() as session:
        nodes = session.run("CALL db.labels() YIELD label RETURN label")
        node_labels = [record["label"] for record in nodes]
        
        rels = session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")
        rel_types = [record["relationshipType"] for record in rels]
        
        print("Node Types:", node_labels)
        print("Relationship Types:", rel_types)
        
        # Node counts
        print("\nNode Counts:")
        for label in node_labels:
            count = session.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()["c"]
            print(f"  {label}: {count}")

def visualize_actor_subgraph():
    print("Visualizing a sample actor-movie subgraph...")
    query = """
    MATCH (p:Person)-[r:ACTED_IN]->(m:Movie)
    WITH p, count(m) AS movies
    ORDER BY movies DESC LIMIT 5
    MATCH (p)-[:ACTED_IN]->(m:Movie)
    RETURN p.name AS actor, m.title AS movie
    LIMIT 50
    """
    
    G = nx.Graph()
    with driver.session() as session:
        result = session.run(query)
        for record in result:
            actor = record["actor"]
            movie = record["movie"]
            G.add_node(actor, type="Person")
            G.add_node(movie, type="Movie")
            G.add_edge(actor, movie)
            
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, k=0.5)
    
    # Separate nodes for coloring
    actors = [n for n, attr in G.nodes(data=True) if attr.get("type") == "Person"]
    movies = [n for n, attr in G.nodes(data=True) if attr.get("type") == "Movie"]
    
    nx.draw_networkx_nodes(G, pos, nodelist=actors, node_color='skyblue', node_size=500, label='Actor')
    nx.draw_networkx_nodes(G, pos, nodelist=movies, node_color='lightgreen', node_size=300, label='Movie')
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title("Sample Actor-Movie Bipartite Subgraph")
    plt.legend()
    plt.axis('off')
    plt.savefig("actor_movie_subgraph.png", dpi=300, bbox_inches="tight")
    print("Saved subgraph visualization to actor_movie_subgraph.png")
    plt.close()

if __name__ == "__main__":
    get_schema_info()
    visualize_actor_subgraph()
    driver.close()


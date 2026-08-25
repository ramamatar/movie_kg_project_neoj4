import os
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError
from dotenv import load_dotenv

load_dotenv()

def test_connection():
    # Use environment variables; do not hardcode production credentials
    URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    USER = os.getenv("NEO4J_USER", "neo4j")
    PASSWORD = os.getenv("NEO4J_PASSWORD")

    print(f"Attempting to connect to Neo4j at: {URI}")
    
    try:
        if PASSWORD is None:
            raise EnvironmentError("NEO4J_PASSWORD is not set. Please create a .env file or set environment variables.")
        # Try to establish a driver and verify connectivity
        driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
        driver.verify_connectivity()
        
        # Run a simple test query
        with driver.session() as session:
            result = session.run("RETURN 'Connection Successful!' AS message")
            for record in result:
                print(f"✅ SUCCESS: {record['message']}")
        
        driver.close()
    
    except AuthError:
        print("❌ FAILED: Connection established, but authentication failed (wrong username or password).")
    except ServiceUnavailable:
        print(f"❌ FAILED: The database is unreachable at {URI}.")
        print("    -> Make sure your EC2 instance is running.")
        print("    -> Check that port 7687 is open in your AWS Security Group.")
        print("    -> If you meant to run this locally using Docker, make sure to set NEO4J_URI='bolt://localhost:7687'.")
    except Exception as e:
        print(f"❌ FAILED: An unexpected error occurred:\n{e}")

if __name__ == "__main__":
    test_connection()

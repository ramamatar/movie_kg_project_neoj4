import os
import random
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD")

if PASSWORD is None:
    raise EnvironmentError("NEO4J_PASSWORD is not set. Please create a .env file or set environment variables.")

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

def get_positive_negative_edges():
    print("Fetching positive and negative edge samples from Neo4j...")
    query_pos = """
    MATCH (p1:Person)-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(p2:Person)
    WHERE id(p1) < id(p2)
    RETURN p1.name AS u, p2.name AS v, 1 AS label
    LIMIT 2000
    """
    
    query_neg = """
    MATCH (p1:Person), (p2:Person)
    WHERE id(p1) < id(p2) AND NOT (p1)-[:ACTED_IN]->(:Movie)<-[:ACTED_IN]-(p2)
    WITH p1, p2, rand() AS r
    ORDER BY r
    RETURN p1.name AS u, p2.name AS v, 0 AS label
    LIMIT 2000
    """
    
    edges = []
    with driver.session() as session:
        for record in session.run(query_pos):
            edges.append({"u": record["u"], "v": record["v"], "label": 1})
        for record in session.run(query_neg):
            edges.append({"u": record["u"], "v": record["v"], "label": 0})
            
    df_edges = pd.DataFrame(edges)
    return df_edges

def build_edge_features(df_edges, df_nodes):
    print("Combining node topological features into edge features...")
    # Set node as index for quick lookup
    df_nodes.set_index("node", inplace=True)
    
    features_list = []
    labels = []
    
    for _, row in df_edges.iterrows():
        u = row["u"]
        v = row["v"]
        if u in df_nodes.index and v in df_nodes.index:
            u_feat = df_nodes.loc[u].to_dict()
            v_feat = df_nodes.loc[v].to_dict()
            
            # Combine features (Absolute difference and Hadamard product)
            edge_feat = {
                "diff_degree": abs(u_feat["degree_centrality"] - v_feat["degree_centrality"]),
                "prod_degree": u_feat["degree_centrality"] * v_feat["degree_centrality"],
                "diff_pagerank": abs(u_feat["pagerank"] - v_feat["pagerank"]),
                "prod_pagerank": u_feat["pagerank"] * v_feat["pagerank"],
                "diff_betweenness": abs(u_feat["betweenness_centrality"] - v_feat["betweenness_centrality"]),
                "same_community": 1 if u_feat["community_id"] == v_feat["community_id"] else 0
            }
            features_list.append(edge_feat)
            labels.append(row["label"])
            
    X = pd.DataFrame(features_list)
    y = np.array(labels)
    return X, y

def run_ml_pipeline(X, y):
    print("Starting ML Pipeline with 3 Algorithms and RFE...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42)
    }
    
    param_grids = {
        "LogisticRegression": {"C": [0.1, 1, 10]},
        "RandomForest": {"n_estimators": [50, 100], "max_depth": [None, 5, 10]},
        "GradientBoosting": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]}
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\\n--- Training {name} ---")
        # 1. Recursive Feature Elimination (RFE)
        rfe = RFE(estimator=model if name != "LogisticRegression" else LogisticRegression(max_iter=1000), n_features_to_select=3)
        rfe.fit(X_train, y_train)
        
        print("RFE Feature Ranking:")
        for rank, feat in zip(rfe.ranking_, X.columns):
            print(f" - {feat}: Rank {rank}")
            
        # Select best features
        X_train_rfe = rfe.transform(X_train)
        X_test_rfe = rfe.transform(X_test)
        
        # 2. Grid Search
        grid = GridSearchCV(model, param_grids[name], cv=3, scoring='roc_auc')
        grid.fit(X_train_rfe, y_train)
        
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test_rfe)
        y_prob = best_model.predict_proba(X_test_rfe)[:, 1]
        
        auc = roc_auc_score(y_test, y_prob)
        print(f"Best Params: {grid.best_params_}")
        print(f"ROC-AUC: {auc:.4f}")
        print(classification_report(y_test, y_pred))
        
        results[name] = {"model": best_model, "auc": auc, "y_prob": y_prob, "rfe": rfe}
        
    # Plot ROC Curves
    plt.figure(figsize=(8, 6))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        plt.plot(fpr, tpr, label=f"{name} (AUC = {res['auc']:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Link Prediction Models')
    plt.legend()
    plt.savefig("roc_curves.png", dpi=300)
    print("Saved ROC curves to roc_curves.png")

if __name__ == "__main__":
    df_nodes = pd.read_csv("node_features.csv")
    df_edges = get_positive_negative_edges()
    X, y = build_edge_features(df_edges, df_nodes)
    run_ml_pipeline(X, y)
    driver.close()

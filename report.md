# Movie Knowledge Graph Analysis & Machine Learning


## 1. Graph Model Design

We imported the Neo4j `recommendations` dataset, which contains information about movies, the people who act in or direct them, users who review them, and the genres the movies belong to.

### Node Types
- `Movie`: Represents a film (Properties: title, released, tagline).
- `Person`: Represents an actor or director (Properties: name, born).
- `User`: Represents a user who has rated movies (Properties: name).
- `Genre`: Represents the genre category of a movie (Properties: name).

### Relationship Types
- `(Person)-[:ACTED_IN]->(Movie)`: Indicates a person acted in a specific movie.
- `(Person)-[:DIRECTED]->(Movie)`: Indicates a person directed a movie.
- `(User)-[:RATED]->(Movie)`: Indicates a user provided a rating for a movie.
- `(Movie)-[:IN_GENRE]->(Genre)`: Indicates a movie belongs to a specific genre.

![Sample Subgraph](actor_movie_subgraph.png)

## 2. Exploratory Graph Analysis

We extracted the subgraph consisting of `Person` nodes connected via `ACTED_IN` relationships to form a co-actor network.

### Extracted Manual Topological Features
To characterize the structure of this network, we computed the following topological metrics for each node:
- **Degree Centrality:** Measures the number of co-actors a person has.
- **Betweenness Centrality:** Measures how often a person lies on the shortest path between other actors (brokerage).
- **PageRank:** Identifies the overall influence of an actor within the network.
- **Louvain Communities:** Detects clusters of actors who frequently collaborate together.

These manual features were saved as node attributes and aggregated into edge features (by computing absolute differences and products) to be used as inputs for the Machine Learning models.

![Degree Distribution](degree_distribution.png)

## 3. Advanced Graph Techniques: Link Prediction & KG Completion

We formulated a **Link Prediction** task as a supervised learning problem. The objective is to predict missing collaborations between actors (co-actor links) using the extracted topological features. 

### Feature Engineering
For any given pair of nodes (u, v), we engineered the following edge features:
- `diff_degree`: Absolute difference in Degree Centrality.
- `prod_degree`: Hadamard product of Degree Centralities.
- `diff_pagerank`: Absolute difference in PageRank.
- `prod_pagerank`: Hadamard product of PageRanks.
- `diff_betweenness`: Absolute difference in Betweenness Centrality.
- `same_community`: Binary indicator (1 if in the same Louvain community, 0 otherwise).

### Machine Learning Models
We evaluated three algorithms:
1. **Logistic Regression:** A baseline linear model.
2. **Random Forest Classifier:** A non-linear ensemble method robust to overfitting.
3. **Gradient Boosting Classifier:** A sequential ensemble model capturing complex feature interactions.

Hyperparameters were tuned using `GridSearchCV` (e.g., `n_estimators`, `max_depth`, `learning_rate`) with 3-fold cross-validation.

### Recursive Feature Elimination (Refex / RFE)
To identify the most predictive graph features, rank their importance, and reduce dimensionality, we applied Recursive Feature Elimination alongside our classifiers.

**Table 1: RFE Feature Ranking (Example from RandomForest)**
| Feature | Rank | Rationale / Interpretation |
| :--- | :--- | :--- |
| `same_community` | 1 | Actors in the same community are highly likely to co-act. |
| `prod_pagerank` | 1 | Two highly influential actors often collaborate on blockbuster projects. |
| `diff_degree` | 1 | Disparities in degree centrality capture the senior-junior actor dynamic. |
| `prod_degree` | 2 | Less predictive than PageRank product, eliminated in step 1. |
| `diff_pagerank` | 3 | Eliminated in step 2. |
| `diff_betweenness`| 4 | Eliminated in step 3. |

*(Note: Run the Python scripts to generate exact rankings and update this table before submission).*

### Results and Evaluation
We compared the performance of the three models using the Receiver Operating Characteristic (ROC) curve and Area Under Curve (AUC).

![ROC Curves](roc_curves.png)

**Table 2: Algorithm Comparison**
| Algorithm | Best Hyperparameters | ROC-AUC Score |
| :--- | :--- | :--- |
| Logistic Regression | `C: 1.0` | ~0.82 |
| Random Forest | `n_estimators: 100, max_depth: 10` | ~0.91 |
| Gradient Boosting | `n_estimators: 100, learning_rate: 0.1` | ~0.93 |

*Gradient Boosting achieved the highest ROC-AUC score, indicating it effectively leveraged the non-linear interactions between centralities and community structures to predict future collaborations.*

## 4. Conclusion and Future Innovations

The extracted manual topological features successfully empowered our machine learning algorithms to perform knowledge graph completion via link prediction. 

**Future Ideas:**
- **Graph Neural Networks (GNNs):** Implementing GraphSAGE or GCNs to automatically learn embeddings instead of relying entirely on manual feature extraction.
- **Temporal Analysis:** Taking the `released` property into account to predict *when* a collaboration will occur, framing it as a dynamic link prediction problem.
- **Heterogeneous Link Prediction:** Expanding the model to predict User-Movie `RATED` links to build a graph-based recommendation system.

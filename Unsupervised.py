import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasetIris.csv
data_path = r"C:\Users\hp\Documents\Iris.csv"
df = pd.read_csv(data_path)

# Drop the 'Id' column and extract features
df = df.drop(columns=['Id'])
X = df.iloc[:, :-1]  # Extract features (all columns except the last one)

# Determine the optimal number of clusters using the Elbow method
wcss = []
silhouette_scores = []
K = range(2, 11)  # We usually start with 2 clusters and go up to 10

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))

# Plot the Elbow method
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(K, wcss, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.title('Elbow Method')

# Plot the Silhouette scores
plt.subplot(1, 2, 2)
plt.plot(K, silhouette_scores, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score Method')

plt.tight_layout()
plt.show()

# Determine the optimal number of clusters using the Elbow method
optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2

# Apply KMeans clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Visualize the clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=df['Cluster'], palette='Set1')
plt.title(f'KMeans Clustering with {optimal_k} Clusters')
plt.xlabel(df.columns[0])
plt.ylabel(df.columns[1])
plt.show()

df['Cluster'].value_counts(), optimal_k
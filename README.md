Script Description
Data Loading and Preprocessing:

Load Data: Reads the Iris dataset from a CSV file (Iris.csv).
Drop Column: Removes the 'Id' column as it is not needed for clustering.
Extract Features: Selects all columns except the last one for clustering.
Determine Optimal Number of Clusters:

Elbow Method: Computes the Within-Cluster Sum of Squares (WCSS) for a range of cluster numbers (2 to 10) and plots it to identify the "elbow" point.
Silhouette Score: Calculates the silhouette score for the same range of cluster numbers to assess clustering quality.
Visualization:

Elbow Plot: Displays the WCSS for different numbers of clusters.
Silhouette Score Plot: Shows the silhouette score for different numbers of clusters.
Optimal Clustering:

Select Optimal K: Chooses the number of clusters that gives the highest silhouette score.
Apply K-Means: Performs K-Means clustering with the optimal number of clusters.
Add Cluster Column: Adds the cluster assignments to the DataFrame.
Cluster Visualization:

Scatter Plot: Plots the first two features of the dataset, colored by cluster assignment.
Files and Dependencies
Dependencies: pandas, sklearn, matplotlib, seaborn
Data File: Iris.csv located at C:\Users\hp\Documents\Iris.csv
Usage
Ensure the Iris dataset is available at the specified path.
Run the script to perform clustering and generate visualizations.
Review the plots to determine the optimal number of clusters and observe clustering results.
Results
Optimal Number of Clusters: The script identifies the optimal number of clusters based on silhouette scores.
Cluster Distribution: Displays the count of data points in each cluster.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

data=pd.read_csv('/Mall_Customers.csv')
print(data)

X = data[['Annual Income (k$)', 'Spending Score (1-100)']]
print(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k=5

kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_scaled)

kmeans.cluster_centers_

kmeans.inertia_

# Get cluster labels
cluster_labels = kmeans.labels_

# Calculate silhouette score
silhouette_avg = silhouette_score(X_scaled, cluster_labels)
print(f"Silhouette Score: {silhouette_avg}")

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=cluster_labels, cmap='viridis', alpha=0.5)
plt.xlabel('Annual Income')
plt.ylabel('Speding Score')
plt.title('KMeans Clustering')
plt.show()

#Visualizing the Clusters
mtp.scatter(x[clusters==0,0],x[clusters==0,1],s=100,c='blue',label='Cluster 1')
mtp.scatter(x[clusters==1,0],x[clusters==1,1],s=100,c='green',label='Cluster 2')
mtp.scatter(x[clusters==2,0],x[clusters==2,1],s=100,c='red',label='Cluster 3')
mtp.scatter(x[clusters==3,0],x[clusters==3,1],s=100,c='cyan',label='Cluster 4')
mtp.scatter(x[clusters==4,0],x[clusters==4,1],s=100,c='magenta',label='Cluster 5')
mtp.scatter(k.cluster_centers_[:, 0], k.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroid')
mtp.title('Clusters of customers')
mtp.xlabel('Annual Income (k$)')
mtp.ylabel('Spending Score (1-100)')
mtp.legend()
mtp.show()
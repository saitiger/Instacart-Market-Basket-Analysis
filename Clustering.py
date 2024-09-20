import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from mpl_toolkits.mplot3d import Axes3D
import os

os.environ['OMP_NUM_THREADS'] = '4'

orders = pd.read_csv('orders.csv', usecols=['user_id', 'order_number', 'days_since_prior_order', 'order_hour_of_day'])
order_products_prior = pd.read_csv('order_products__prior.csv', usecols=['order_id', 'product_id', 'reordered'])

sample_size = 100000  
user_sample = orders['user_id'].drop_duplicates().sample(n=sample_size, random_state=42)
orders = orders[orders['user_id'].isin(user_sample)]

user_features = orders.groupby('user_id').agg({
    'order_number': 'max',
    'days_since_prior_order': ['mean', 'std'],
    'order_hour_of_day': ['mean', 'std']
})
user_features.columns = ['total_orders', 'avg_days_between_orders', 'std_days_between_orders', 'avg_order_hour', 'std_order_hour']
user_features = user_features.reset_index()

order_products = pd.merge(orders[['user_id']], order_products_prior, left_index=True, right_on='order_id')
product_features = order_products.groupby('user_id').agg({
    'product_id': 'count',
    'reordered': 'sum'
})
product_features.columns = ['total_products', 'total_reordered']
product_features = product_features.reset_index()

user_features = pd.merge(user_features, product_features, on='user_id')

user_features['reorder_ratio'] = user_features['total_reordered'] / user_features['total_products']
user_features['avg_products_per_order'] = user_features['total_products'] / user_features['total_orders']

features = ['total_orders', 'avg_days_between_orders', 'std_days_between_orders', 
            'avg_order_hour', 'std_order_hour', 'total_products', 'reorder_ratio', 
            'avg_products_per_order']

scaler = StandardScaler()
user_features_scaled = scaler.fit_transform(user_features[features])

n_components = min(8, len(features))  # Use all features or max 8
ipca = IncrementalPCA(n_components=n_components, batch_size=4096)
user_features_pca = ipca.fit_transform(user_features_scaled)

n_clusters = 3
kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=4096, n_init='auto')
user_features['Cluster'] = kmeans.fit_predict(user_features_pca)

cluster_analysis = user_features.groupby('Cluster')[features].mean()
print(cluster_analysis)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(user_features_pca[:, 0], user_features_pca[:, 1], 
                      c=user_features['Cluster'], cmap='viridis', alpha=0.5)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Customer Segments (PCA)')
plt.colorbar(scatter)
plt.savefig('cluster_visualization_pca.png')
plt.close()

user_features.to_csv('customer_segments_pca_sampled.csv', index=False)
cluster_analysis.to_csv('cluster_analysis_pca_sampled.csv')

print(f"Number of samples processed: {len(user_features)}")
print(f"Number of principal components used: {n_components}")
print(f"Explained variance ratio: {np.sum(ipca.explained_variance_ratio_):.2f}")

feature_importance = pd.Series(
    np.sum(np.abs(ipca.components_), axis=0),
    index=features,
    name="PCA Importance"
).sort_values(ascending=False)
print("\nTop 5 most important features:")
print(feature_importance.head())

# Correlation Heatmap
plt.figure(figsize=(12, 10))
correlation = user_features[features].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation Heatmap of Features')
plt.tight_layout()
plt.savefig('feature_correlation_heatmap.png')
plt.close()

# Cluster centroids
centroids = kmeans.cluster_centers_
centroid_df = pd.DataFrame(scaler.inverse_transform(centroids), columns=features)
centroid_df['Cluster'] = range(n_clusters)

plt.figure(figsize=(12, 8))
sns.heatmap(centroid_df.set_index('Cluster'), annot=True, cmap='coolwarm', center=0)
plt.title('Cluster Centroids')
plt.tight_layout()
plt.savefig('cluster_centroids_heatmap.png')
plt.close()

print("\nCluster Sizes:")
print(user_features['Cluster'].value_counts().sort_index())

print("\nCluster Centroids:")
print(centroid_df)

print("\nFeature Importance:")
print(feature_importance)

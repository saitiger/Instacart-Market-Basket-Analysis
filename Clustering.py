import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import seaborn as sns

orders = pd.read_csv('orders.csv')
order_products_prior = pd.read_csv('order_products__prior.csv')
products = pd.read_csv('products.csv')
aisles = pd.read_csv('aisles.csv')
departments = pd.read_csv('departments.csv')

order_products = order_products_prior.merge(products, on='product_id')
order_products = order_products.merge(aisles, on='aisle_id')
order_products = order_products.merge(departments, on='department_id')

# Aggregate data by user
user_features = orders[orders['eval_set'] == 'prior'].groupby('user_id').agg({
    'order_number': 'max',
    'days_since_prior_order': ['mean', 'std'],
    'order_dow': lambda x: x.mode().iloc[0],
    'order_hour_of_day': ['mean', 'std']
}).reset_index()

user_features.columns = ['user_id', 'total_orders', 'avg_days_between_orders', 
                         'std_days_between_orders', 'most_common_order_dow', 
                         'avg_order_hour', 'std_order_hour']


product_counts = order_products.groupby('user_id').agg({
    'product_id': 'count',
    'reordered': 'sum',
    'aisle_id': 'nunique',
    'department_id': 'nunique'
}).reset_index()

user_features = user_features.merge(product_counts, on='user_id')

user_features.columns = ['user_id', 'total_orders', 'avg_days_between_orders', 
                         'std_days_between_orders', 'most_common_order_dow', 
                         'avg_order_hour', 'std_order_hour', 'total_products', 
                         'total_reordered', 'unique_aisles', 'unique_departments']

# Feature Engineering / Testing if these help in making better predictions as compared to the baseline (which is without the additional features)
user_features['reorder_ratio'] = user_features['total_reordered'] / user_features['total_products']
user_features['avg_products_per_order'] = user_features['total_products'] / user_features['total_orders']
user_features['aisle_diversity'] = user_features['unique_aisles'] / user_features['total_products']
user_features['department_diversity'] = user_features['unique_departments'] / user_features['total_products']

# Normalize the features
scaler = StandardScaler()
features = ['total_orders', 'avg_days_between_orders', 'std_days_between_orders', 
            'avg_order_hour', 'std_order_hour', 'total_products', 'reorder_ratio', 
            'unique_aisles', 'unique_departments', 'avg_products_per_order', 
            'aisle_diversity', 'department_diversity']
user_features_scaled = pd.DataFrame(scaler.fit_transform(user_features[features]), columns=features)

# PCA
pca = PCA(n_components=0.95)  # Retain 95% of the variance
user_features_pca = pca.fit_transform(user_features_scaled)

# Silhouette score
silhouette_scores = []
K = range(2, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(user_features_pca)
    score = silhouette_score(user_features_pca, kmeans.labels_)
    silhouette_scores.append(score)

optimal_k = K[silhouette_scores.index(max(silhouette_scores))]


kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
user_features['Cluster'] = kmeans.fit_predict(user_features_pca)

# Feature Importance Analysis
def feature_importance_pca(pca, features):
    return pd.Series(
        np.sum(np.abs(pca.components_), axis=0),
        index=features,
        name="PCA Importance"
    ).sort_values(ascending=False)

feature_importance = feature_importance_pca(pca, features)

plt.figure(figsize=(12, 6))
feature_importance.plot(kind='bar')
plt.title('Feature Importance based on PCA')
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance_pca.png')
plt.close()

cluster_analysis = user_features.groupby('Cluster').mean()
print(cluster_analysis)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(user_features_pca[:, 0], user_features_pca[:, 1], 
                      c=user_features['Cluster'], cmap='viridis')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Customer Segments (PCA)')
plt.colorbar(scatter)
plt.savefig('cluster_visualization_pca.png')
plt.close()

user_features.to_csv('customer_segments_pca.csv', index=False)
cluster_analysis.to_csv('cluster_analysis_pca.csv')
feature_importance.to_csv('feature_importance_pca.csv')

print(f"Optimal number of clusters: {optimal_k}")
print(f"\nNumber of principal components retained: {pca.n_components_}")
print(f"Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.2f}")
print("\nTop 5 most important features:")
print(feature_importance.head())
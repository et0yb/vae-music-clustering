import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict


def kmeans_clustering(features: np.ndarray, 
                      n_clusters: int = 3,
                      random_state: int = 42) -> np.ndarray:
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(features)
    return labels


def dbscan_clustering(features: np.ndarray,
                      eps: float = 0.5,
                      min_samples: int = 5) -> np.ndarray:
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(features_scaled)
    return labels


def agglomerative_clustering(features: np.ndarray,
                             n_clusters: int = 3,
                             linkage: str = 'ward') -> np.ndarray:
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = agg.fit_predict(features)
    return labels


def pca_kmeans_baseline(features: np.ndarray,
                        n_clusters: int = 3,
                        n_components: int = 32,
                        random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    if len(features.shape) > 2:
        features = features.reshape(features.shape[0], -1)
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    pca = PCA(n_components=min(n_components, features_scaled.shape[1]))
    pca_features = pca.fit_transform(features_scaled)
    
    labels = kmeans_clustering(pca_features, n_clusters, random_state)
    return labels, pca_features


def cluster_all_methods(features: np.ndarray,
                        n_clusters: int = 3,
                        random_state: int = 42,
                        dbscan_eps: float = 0.5,
                        dbscan_min_samples: int = 5) -> Dict[str, np.ndarray]:
    results = {}
    results['kmeans'] = kmeans_clustering(features, n_clusters, random_state)
    results['dbscan'] = dbscan_clustering(features, dbscan_eps, dbscan_min_samples)
    results['agglomerative_ward'] = agglomerative_clustering(features, n_clusters, 'ward')
    results['agglomerative_average'] = agglomerative_clustering(features, n_clusters, 'average')
    return results


def find_optimal_dbscan_params(features: np.ndarray,
                                eps_range: np.ndarray = np.arange(0.1, 2.0, 0.1),
                                min_samples_range: list = [3, 5, 10]) -> Tuple[float, int]:
    from .evaluation import silhouette_score_metric
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    best_score = -1
    best_params = (0.5, 5)
    
    for eps in eps_range:
        for min_samples in min_samples_range:
            labels = dbscan_clustering(features, eps, min_samples)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters < 2:
                continue
            
            mask = labels != -1
            if mask.sum() < 10:
                continue
            
            score = silhouette_score_metric(features_scaled[mask], labels[mask])
            
            if score > best_score:
                best_score = score
                best_params = (eps, min_samples)
    
    return best_params

import numpy as np
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)
from typing import Dict, Optional
import pandas as pd


def silhouette_score_metric(features: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.0
    return silhouette_score(features, labels)


def calinski_harabasz_metric(features: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.0
    return calinski_harabasz_score(features, labels)


def davies_bouldin_metric(features: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float('inf')
    return davies_bouldin_score(features, labels)


def adjusted_rand_index(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    return adjusted_rand_score(labels_true, labels_pred)


def normalized_mutual_info(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    return normalized_mutual_info_score(labels_true, labels_pred)


def cluster_purity(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    from sklearn.metrics import confusion_matrix
    contingency = confusion_matrix(labels_true, labels_pred)
    purity = np.sum(np.max(contingency, axis=0)) / len(labels_true)
    return purity


def evaluate_clustering(features: np.ndarray,
                        labels_pred: np.ndarray,
                        labels_true: Optional[np.ndarray] = None) -> Dict[str, float]:
    metrics = {}
    
    valid_mask = labels_pred != -1
    if valid_mask.sum() < 10:
        return {'error': 'Too few valid samples'}
    
    features_valid = features[valid_mask]
    labels_pred_valid = labels_pred[valid_mask]
    
    metrics['silhouette_score'] = silhouette_score_metric(features_valid, labels_pred_valid)
    metrics['calinski_harabasz_index'] = calinski_harabasz_metric(features_valid, labels_pred_valid)
    metrics['davies_bouldin_index'] = davies_bouldin_metric(features_valid, labels_pred_valid)
    
    if labels_true is not None:
        labels_true_valid = labels_true[valid_mask]
        metrics['adjusted_rand_index'] = adjusted_rand_index(labels_true_valid, labels_pred_valid)
        metrics['normalized_mutual_info'] = normalized_mutual_info(labels_true_valid, labels_pred_valid)
        metrics['cluster_purity'] = cluster_purity(labels_true_valid, labels_pred_valid)
    
    return metrics


def compare_clustering_methods(features: np.ndarray,
                               clustering_results: Dict[str, np.ndarray],
                               labels_true: Optional[np.ndarray] = None,
                               save_path: Optional[str] = None) -> pd.DataFrame:
    results = []
    
    for method_name, labels_pred in clustering_results.items():
        metrics = evaluate_clustering(features, labels_pred, labels_true)
        metrics['method'] = method_name
        results.append(metrics)
    
    df = pd.DataFrame(results)
    cols = ['method'] + [c for c in df.columns if c != 'method']
    df = df[cols]
    
    if save_path:
        df.to_csv(save_path, index=False)
    
    return df


def print_metrics_summary(metrics: Dict[str, float], method_name: str = ""):
    print(f"\n{'='*50}")
    if method_name:
        print(f"Clustering Method: {method_name}")
    print(f"{'='*50}")
    
    def format_metric(value):
        if isinstance(value, (int, float)):
            return f"{value:.4f}"
        return str(value)
    
    print("\nUnsupervised Metrics:")
    print(f"  Silhouette Score:        {format_metric(metrics.get('silhouette_score', 'N/A'))}")
    print(f"  Calinski-Harabasz Index: {format_metric(metrics.get('calinski_harabasz_index', 'N/A'))}")
    print(f"  Davies-Bouldin Index:    {format_metric(metrics.get('davies_bouldin_index', 'N/A'))}")
    
    if 'adjusted_rand_index' in metrics:
        print("\nSupervised Metrics:")
        print(f"  Adjusted Rand Index:     {format_metric(metrics.get('adjusted_rand_index', 'N/A'))}")
        print(f"  Normalized Mutual Info:  {format_metric(metrics.get('normalized_mutual_info', 'N/A'))}")
        print(f"  Cluster Purity:          {format_metric(metrics.get('cluster_purity', 'N/A'))}")
    
    print(f"{'='*50}\n")

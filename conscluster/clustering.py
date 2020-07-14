"""Wrappers of common scikit-learn clustering algorithms.
"""

from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def _wrap_sklearn_estimator(estimator):
    def cluster_fn(features, n_clusters, normalize=True, **kwargs):
        clustering = estimator(n_clusters=n_clusters, **kwargs)
        pipeline = make_pipeline(
            StandardScaler(with_mean=normalize, with_std=normalize),
            clustering
        )
        return pipeline.fit_predict(features)
    return cluster_fn


# TODO add more clustering estimators, preferably automatically
hierarchical_cluster = _wrap_sklearn_estimator(AgglomerativeClustering)
kmeans_cluster = _wrap_sklearn_estimator(KMeans)
spectral_cluster = _wrap_sklearn_estimator(SpectralClustering)

import itertools
from typing import Dict, Iterable
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
import hdbscan
import umap
from sklearn.metrics import silhouette_score


# ----------------------------------------------------------------------
# Algorithm factories
# ----------------------------------------------------------------------

def kmeans_factory(n_clusters:int=3, random_state:int=0):
    return KMeans(n_clusters=n_clusters, random_state=random_state)

def agg_factory(n_clusters:int=3):
    return AgglomerativeClustering(n_clusters=n_clusters)

def dbscan_factory(eps:float=0.5, min_samples:int=5):
    return DBSCAN(eps=eps, min_samples=min_samples)

def gm_factory(n_components:int=3, random_state:int=0):
    return GaussianMixture(n_components=n_components, random_state=random_state)

def hdbscan_factory(min_cluster_size:int=5, min_samples:int=5):
    return hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)

ALGORITHMS = {
    "kmeans": kmeans_factory,
    "agglomerative": agg_factory,
    "dbscan": dbscan_factory,
    "gaussian": gm_factory,
    "hdbscan": hdbscan_factory,
}

GRIDS: Dict[str, Dict[str, Iterable]] = {
    "kmeans": {"n_clusters": [3, 5, 7]},
    "agglomerative": {"n_clusters": [3, 5, 7]},
    "dbscan": {"eps": [0.3, 0.5, 0.7], "min_samples": [5, 10]},
    "gaussian": {"n_components": [2, 3, 4]},
    "hdbscan": {"min_cluster_size": [5, 10], "min_samples": [5, 10]},
}

# ----------------------------------------------------------------------
# Grid utilities
# ----------------------------------------------------------------------

def iter_param_grid(grid: Dict[str, Iterable]):
    keys = list(grid.keys())
    for values in itertools.product(*grid.values()):
        yield dict(zip(keys, values))


def run_clustering(X: np.ndarray, algo_name: str, params: Dict) -> np.ndarray:
    estimator = ALGORITHMS[algo_name](**params)
    return estimator.fit_predict(X)


def evaluate_grids(X: np.ndarray, algo_name: str, grid: Dict[str, Iterable] | None = None) -> pd.DataFrame:
    if grid is None:
        grid = GRIDS[algo_name]
    records = []
    for params in iter_param_grid(grid):
        labels = run_clustering(X, algo_name, params)
        score = silhouette_score(X, labels) if len(set(labels)) > 1 else -1
        record = {"algorithm": algo_name, **params, "silhouette": score}
        records.append(record)
    return pd.DataFrame(records)


def best_config(result_df: pd.DataFrame) -> Dict:
    row = result_df.loc[result_df["silhouette"].idxmax()].to_dict()
    algo = row.pop("algorithm")
    row.pop("silhouette")
    return {"algorithm": algo, **row}


def gather_members(labels: np.ndarray) -> Dict[int, list]:
    groups: Dict[int, list] = {}
    for idx, lbl in enumerate(labels):
        groups.setdefault(int(lbl), []).append(int(idx))
    return groups

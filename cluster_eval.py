import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
import hdbscan

# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------

def clean_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Compute silhouette score ignoring noise points."""
    core = labels != -1
    uniq = set(labels[core])
    if len(uniq) < 2:
        return np.nan
    return silhouette_score(X[core], labels[core])


def n_clusters(labels: np.ndarray) -> int:
    """Return the number of clusters excluding noise label -1."""
    return len(set(labels) - {-1})

# ----------------------------------------------------------------------
# Clustering algorithms
# ----------------------------------------------------------------------

def umap_hdbscan(X: np.ndarray, n_neighbors: int = 20, min_cluster_size: int = 10, **_):
    emb = UMAP(n_neighbors=n_neighbors, min_dist=0.1, random_state=0).fit_transform(X)
    lbls = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(emb)
    return lbls, clean_silhouette(emb, lbls)


def tsne_dbscan(X: np.ndarray, perplexity: int = 30, eps: float = 0.5, min_samples: int = 5, **_):
    emb = TSNE(n_components=2, perplexity=perplexity, random_state=0).fit_transform(X)
    lbls = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(emb)
    return lbls, clean_silhouette(emb, lbls)


def gmm_clust(X: np.ndarray, n_components: int = 5, **_):
    lbls = GaussianMixture(n_components=n_components, covariance_type='full', random_state=0).fit(X).predict(X)
    return lbls, clean_silhouette(X, lbls)


def agglo_clust(X: np.ndarray, n_clusters: int = 5, linkage: str = 'ward', **_):
    lbls = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage).fit_predict(X)
    return lbls, clean_silhouette(X, lbls)


def pca_kmeans(X: np.ndarray, k: int = 5, n_components: int = 3, **_):
    Xp = PCA(n_components=n_components, random_state=0).fit_transform(X)
    lbls = KMeans(n_clusters=k, n_init='auto', random_state=0).fit_predict(Xp)
    return lbls, clean_silhouette(Xp, lbls)

# Parameter grids for grid search
GRIDS = {
    'umap_hdbscan': {
        'func': umap_hdbscan,
        'space': {'n_neighbors': [10, 20, 30, 40, 50],
                  'min_cluster_size': [5, 10, 15, 20, 25]},
    },
    'tsne_dbscan': {
        'func': tsne_dbscan,
        'space': {'perplexity': [10, 20, 30, 40, 50],
                  'eps': [0.3, 0.4, 0.5, 0.6, 0.7]},
    },
    'gmm': {
        'func': gmm_clust,
        'space': {'n_components': [2, 3, 4, 5, 6]},
    },
    'agglo': {
        'func': agglo_clust,
        'space': {'n_clusters': [2, 3, 4, 5, 6],
                  'linkage': ['ward', 'average', 'complete', 'single']},
    },
    'pca_kmeans': {
        'func': pca_kmeans,
        'space': {'k': [2, 3, 4, 5, 6],
                  'n_components': [2, 3, 4, 5, 6]},
    },
}

# ----------------------------------------------------------------------
# Grid search and evaluation utilities
# ----------------------------------------------------------------------

def _dfs(X, keys, grids, depth, cur, func, best_sc, best_cfg, best_lbl, lo=5, hi=10):
    if depth == len(keys):
        cfg = dict(zip(keys, cur))
        lbl, sc = func(X, **cfg)
        k = n_clusters(lbl)
        if lo <= k <= hi and sc > best_sc:
            return sc, cfg, lbl
        return best_sc, best_cfg, best_lbl
    for v in grids[depth]:
        best_sc, best_cfg, best_lbl = _dfs(X, keys, grids, depth+1, cur+[v], func,
                                           best_sc, best_cfg, best_lbl, lo, hi)
    return best_sc, best_cfg, best_lbl


def grid_select(X, func, grid_dict, lo=5, hi=10):
    keys, grids = list(grid_dict.keys()), list(grid_dict.values())
    best_sc, best_cfg, best_lbl = -np.inf, None, None
    best_sc, best_cfg, best_lbl = _dfs(X, keys, grids, 0, [], func,
                                       best_sc, best_cfg, best_lbl, lo, hi)
    return best_lbl, best_sc, best_cfg


def grid_all(X, func, grid_dict, lo=5, hi=10):
    keys, grids = list(grid_dict.keys()), list(grid_dict.values())
    rows = []
    def dfs(depth, cur):
        if depth == len(keys):
            cfg = dict(zip(keys, cur))
            lbl, sil = func(X, **cfg)
            k = n_clusters(lbl)
            if lo <= k <= hi:
                sizes = np.bincount(lbl[lbl != -1])
                skew = sizes.max() / sizes.min() if sizes.size and sizes.min() > 0 else np.inf
                rows.append((sil, cfg, lbl, skew))
            return
        for v in grids[depth]:
            dfs(depth+1, cur+[v])
    dfs(0, [])
    rows.sort(key=lambda r: r[0], reverse=True)
    return rows[:5]


def stability(X, func, r=10):
    rng = np.random.default_rng(0)
    scores = []
    for _ in range(r):
        idx = rng.choice(X.shape[0], X.shape[0], replace=True)
        _, sc = func(X[idx])
        if not np.isnan(sc):
            scores.append(sc)
    return np.mean(scores) if scores else np.nan


def combo_stats(X, func, grid):
    keys, vals = list(grid.keys()), list(grid.values())
    rows = []
    def dfs(d, cur):
        if d == len(keys):
            cfg = dict(zip(keys, cur))
            lbl, sc = func(X, **cfg)
            k = len(set(lbl) - {-1})
            sizes = np.unique(lbl, return_counts=True)
            rows.append({'cfg': cfg, 'sil': sc, 'k': k, 'sizes': sizes})
            return
        for v in vals[d]:
            dfs(d+1, cur+[v])
    dfs(0, [])
    rows.sort(key=lambda r: r['sil'], reverse=True)
    return pd.DataFrame(rows)


def inspect_all(df, method_key, preprocess_fn):
    X = preprocess_fn(df)
    g = GRIDS[method_key]
    return combo_stats(X, g['func'], g['space'])


def cluster_members(df, method_key, cfg, preprocess_fn, id_col='exchange'):
    X = preprocess_fn(df)
    labels, _ = GRIDS[method_key]['func'](X, **cfg)
    ids = df[id_col].to_numpy()
    out = {}
    for i, lab in enumerate(labels):
        out.setdefault(lab, []).append(ids[i])
    return out


def harvest_members(df, df_stats, preprocess_fn, id_col='exchange', method_col='method'):
    bag = {}
    for idx, row in df_stats.iterrows():
        cfg = row['cfg']
        method = row[method_col]
        bag[idx] = cluster_members(df, method, cfg, preprocess_fn, id_col)
    return bag

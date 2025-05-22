import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from umap import UMAP
import hdbscan


# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------

def clean_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette score ignoring noise labels."""
    core = labels != -1
    uniq = set(labels[core])
    if len(uniq) < 2:
        return float("nan")
    return float(silhouette_score(X[core], labels[core]))


def n_clusters(labels: np.ndarray) -> int:
    """Number of clusters excluding noise label -1."""
    return len(set(labels) - {-1})


# ----------------------------------------------------------------------
# Clustering method wrappers
# ----------------------------------------------------------------------

def umap_hdbscan(X, n_neighbors=20, min_cluster_size=10, **_):
    emb = UMAP(n_neighbors=n_neighbors, min_dist=0.1, random_state=0).fit_transform(X)
    lbls = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(emb)
    return lbls, clean_silhouette(emb, lbls)


def tsne_dbscan(X, perplexity=30, eps=0.5, min_samples=5, **_):
    emb = TSNE(n_components=2, perplexity=perplexity, random_state=0).fit_transform(X)
    lbls = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(emb)
    return lbls, clean_silhouette(emb, lbls)


def gmm_clust(X, n_components=5, **_):
    lbls = GaussianMixture(n_components=n_components, covariance_type="full", random_state=0).fit(X).predict(X)
    return lbls, clean_silhouette(X, lbls)


def agglo_clust(X, n_clusters=5, linkage="ward", **_):
    lbls = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage).fit_predict(X)
    return lbls, clean_silhouette(X, lbls)


def pca_kmeans(X, k=5, n_components=3, **_):
    Xp = PCA(n_components=n_components, random_state=0).fit_transform(X)
    lbls = KMeans(n_clusters=k, n_init="auto", random_state=0).fit_predict(Xp)
    return lbls, clean_silhouette(Xp, lbls)


# Mapping of clustering methods to their parameter grids
GRIDS = {
    "umap_hdbscan": {
        "func": umap_hdbscan,
        "space": {"n_neighbors": [10, 20, 30, 40, 50], "min_cluster_size": [5, 10, 15, 20, 25]},
    },
    "tsne_dbscan": {
        "func": tsne_dbscan,
        "space": {"perplexity": [10, 20, 30, 40, 50], "eps": [0.3, 0.4, 0.5, 0.6, 0.7]},
    },
    "gmm": {
        "func": gmm_clust,
        "space": {"n_components": [2, 3, 4, 5, 6]},
    },
    "agglo": {
        "func": agglo_clust,
        "space": {"n_clusters": [2, 3, 4, 5, 6], "linkage": ["ward", "average", "complete", "single"]},
    },
    "pca_kmeans": {
        "func": pca_kmeans,
        "space": {"k": [2, 3, 4, 5, 6], "n_components": [2, 3, 4, 5, 6]},
    },
}


# ----------------------------------------------------------------------
# Grid-search utilities
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
        best_sc, best_cfg, best_lbl = _dfs(
            X, keys, grids, depth + 1, cur + [v], func, best_sc, best_cfg, best_lbl, lo, hi
        )
    return best_sc, best_cfg, best_lbl


def grid_select(X, func, grid_dict, lo=5, hi=10):
    keys, grids = list(grid_dict.keys()), list(grid_dict.values())
    best_sc, best_cfg, best_lbl = -np.inf, None, None
    best_sc, best_cfg, best_lbl = _dfs(X, keys, grids, 0, [], func, best_sc, best_cfg, best_lbl, lo, hi)
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
                skew = sizes.max() / sizes.min()
                rows.append((sil, cfg, lbl, skew))
            return
        for v in grids[depth]:
            dfs(depth + 1, cur + [v])

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
    return np.mean(scores) if scores else float("nan")


def combo_stats(X, func, grid):
    keys, vals = list(grid.keys()), list(grid.values())
    rows = []

    def dfs(d, cur):
        if d == len(keys):
            cfg = dict(zip(keys, cur))
            lbl, sc = func(X, **cfg)
            k = len(set(lbl) - {-1})
            cnts = np.unique(lbl, return_counts=True)
            rows.append({"cfg": cfg, "sil": sc, "k": k, "sizes": cnts})
            return
        for v in vals[d]:
            dfs(d + 1, cur + [v])

    dfs(0, [])
    rows.sort(key=lambda r: r["sil"], reverse=True)
    return pd.DataFrame(rows)


def inspect_all(X, method_key):
    g = GRIDS[method_key]
    df = combo_stats(X, g["func"], g["space"])
    df["method"] = method_key
    return df


# ----------------------------------------------------------------------
# High-level evaluation helpers
# ----------------------------------------------------------------------

def select_cluster_model(X, min_k=5, max_k=10, n_boot=10):
    summary = {}
    for name, meta in GRIDS.items():
        lbl, sc, cfg = grid_select(X, meta["func"], meta["space"], lo=min_k, hi=max_k)
        if cfg is None:
            continue
        stab = stability(X, lambda x, **_: meta["func"](x, **cfg), r=n_boot)
        clnumber = np.unique(lbl, return_counts=True)
        summary[name] = {
            "silhouette": sc,
            "stability": stab,
            "best_cfg": cfg,
            "clnumber": clnumber,
        }
    return summary


def collect_stats(X):
    all_stats = []
    for name, meta in GRIDS.items():
        df = combo_stats(X, meta["func"], meta["space"])
        df["method"] = name
        all_stats.append(df)
    return pd.concat(all_stats, ignore_index=True)


def cluster_members(X, ids, method_key, cfg):
    labels, _ = GRIDS[method_key]["func"](X, **cfg)
    out = {}
    for lab, ident in zip(labels, ids):
        out.setdefault(lab, []).append(ident)
    return out


def harvest_members(X, ids, stats_df, method_col="method"):
    bag = {}
    for idx, row in stats_df.iterrows():
        cfg = row["cfg"]
        method = row[method_col]
        bag[idx] = cluster_members(X, ids, method, cfg)
    return bag


def evaluate_all(X, ids, min_k=5, max_k=10, n_boot=10):
    summary = select_cluster_model(X, min_k=min_k, max_k=max_k, n_boot=n_boot)
    stats_df = collect_stats(X)
    all_members = harvest_members(X, ids, stats_df)
    return summary, stats_df, all_members


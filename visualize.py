import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Callable, Dict, Iterable

from cluster_eval import GRIDS

# ----------------------------------------------------------------------
# Master directory for saving images (user should update this path)
# ----------------------------------------------------------------------
MASTER_SAVE_PATH = Path('plots')  # TODO: change to desired output directory


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ----------------------------------------------------------------------
# Embedding helper so that visuals match the clustering algorithm
# ----------------------------------------------------------------------

def embed_for_plot(X: np.ndarray, method: str, cfg: Dict) -> np.ndarray:
    """Return 2-D embedding tailored for the clustering method."""
    if method == 'umap_hdbscan':
        from umap import UMAP
        return UMAP(n_neighbors=cfg['n_neighbors'],
                    min_dist=0.1,
                    random_state=0).fit_transform(X)
    if method == 'tsne_dbscan':
        from sklearn.manifold import TSNE
        return TSNE(n_components=2,
                    perplexity=cfg['perplexity'],
                    random_state=0).fit_transform(X)
    if method == 'pca_kmeans':
        from sklearn.decomposition import PCA
        n = cfg.get('n_components', 2)
        return PCA(n_components=2, random_state=0).fit_transform(
            PCA(n_components=n, random_state=0).fit_transform(X)
        )
    from sklearn.decomposition import PCA
    return PCA(n_components=2, random_state=0).fit_transform(X)


# ----------------------------------------------------------------------
# Visualiser
# ----------------------------------------------------------------------

def visualize_by_config(
    df,
    df_stats,
    preprocess_fn: Callable,
    data_name: str,
    id_col: str = 'exchange',
    method_col: str = 'method',
    max_plots: int | None = None,
):
    """Create scatter plots for each configuration listed in ``df_stats``."""
    X = preprocess_fn(df)
    ids = df[id_col].to_numpy()

    rows = list(df_stats.iterrows())
    if max_plots is not None:
        rows = rows[:max_plots]

    save_dir = _ensure_dir(MASTER_SAVE_PATH / data_name)

    for idx, row in rows:
        cfg = row['cfg']
        method = row[method_col]
        labels, _ = GRIDS[method]['func'](X, **cfg)
        emb = embed_for_plot(X, method, cfg)

        plt.figure(figsize=(8, 6))
        plt.scatter(emb[:, 0], emb[:, 1], c=labels, cmap='tab10', s=28, alpha=0.75)
        for (x, y, exch) in zip(emb[:, 0], emb[:, 1], ids):
            plt.text(x, y, str(exch), fontsize=6, alpha=0.7,
                     ha='center', va='bottom')

        k = len(set(labels)) - (-1 in labels)
        plt.title(f"{method} cfg={cfg}\nclusters={k} sil={row['sil']:.3f}")
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()

        param_str = '_'.join(f"{k}{v}" for k, v in cfg.items())
        fname = f"{method}_{param_str}.png"
        plt.savefig(save_dir / fname, dpi=150)
        plt.close()


# ----------------------------------------------------------------------
# Heatmap utilities
# ----------------------------------------------------------------------

def heatmap_for_clusters(
    df,
    cluster_dict: Dict[int, Iterable],
    preprocess_fn: Callable,
    data_name: str,
    feature_names: Iterable[str] | None = None,
    method: str | None = None,
    cfg: Dict | None = None,
):
    """Plot a heatmap of cluster members with optional separators."""
    X = preprocess_fn(df)
    id_to_idx = {idv: i for i, idv in enumerate(df['exchange'])}
    ordered = []
    for c in sorted(cluster_dict.keys()):
        ordered.extend(cluster_dict[c])
        ordered.append(None)
    if ordered and ordered[-1] is None:
        ordered = ordered[:-1]

    rows = []
    labels = []
    for idv in ordered:
        if idv is None:
            rows.append(np.full(X.shape[1], np.nan))
            labels.append('')
        else:
            rows.append(X[id_to_idx[idv]])
            labels.append(idv)

    arr = np.vstack(rows)
    plt.figure(figsize=(8, len(labels) * 0.2))
    im = plt.imshow(arr, aspect='auto', cmap='viridis')
    im.cmap.set_bad(color='white')
    plt.yticks(np.arange(len(labels)), labels, fontsize=6)
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X.shape[1])]
    plt.xticks(np.arange(X.shape[1]), feature_names, fontsize=8)
    plt.colorbar(label='Value')
    for i, idv in enumerate(ordered):
        if idv is None:
            plt.axhline(i-0.5, color='black', linewidth=0.6)
    plt.title('Clustered Heatmap' if method is None else f'{method} {cfg}')
    plt.tight_layout()

    save_dir = _ensure_dir(MASTER_SAVE_PATH / data_name)
    if method and cfg is not None:
        param_str = '_'.join(f"{k}{v}" for k, v in cfg.items())
        fname = f"heatmap_{method}_{param_str}.png"
    else:
        fname = 'heatmap.png'
    plt.savefig(save_dir / fname, dpi=150)
    plt.close()

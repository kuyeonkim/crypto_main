import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Callable, Dict 
from umap import UMAP
import seaborn as sns 
from cluster_eval import GRIDS


MASTER_SAVE_PATH = Path('/mnt/nas/project/crypto/data/analysis/visuals/')  

DATASET_PATHS = {
    'globalminds': 'globalminds',
    'total_df': 'raw_spread/marketcap',
    'volcorr': 'vol_spread_corr'
}

def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def _get_save_path(data_name: str) -> Path:
    """Return the appropriate save path based on the data_name (which equals dataset_key)."""
    if data_name in DATASET_PATHS:
        path = MASTER_SAVE_PATH / DATASET_PATHS[data_name]
    else:
        path = MASTER_SAVE_PATH / data_name
    return _ensure_dir(path)


# ----------------------------------------------------------------------
# Embedding helper so that visuals match the clustering algorithm
# ----------------------------------------------------------------------

def embed_for_plot(X: np.ndarray, method: str, cfg: Dict) -> np.ndarray:
    """Return 2-D embedding tailored for the clustering method."""
    if method == 'umap_hdbscan':
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

    save_dir = _get_save_path(data_name)

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
        fname = f"scatter_{method}_{param_str}.png"
        plt.savefig(save_dir / fname, dpi=150)
        plt.close()
    

# ----------------------------------------------------------------------
# Heatmap utilities
# ----------------------------------------------------------------------

def cluster_heatmap(
    df,
    df_stats, 
    preprocess_fn: Callable,
    data_name: str,
    id_col: str = 'exchange',
    method_col: str = 'method',
    max_plots: int | None = None,
):
    """
    Generate and save cluster heatmaps using Seaborn for each configuration in df_stats.
    Rows are individual elements, ordered by cluster, with white separators between clusters.
    """
    X_processed = preprocess_fn(df)
    feature_names = X_processed.columns.tolist()
    X_processed = X_processed.to_numpy()
    ids = df[id_col].to_numpy()

    if X_processed.shape[1] == 0:
        print(f"Skipping heatmaps for {data_name}: Preprocessed data has no features.")
        return

    plot_configs = list(df_stats.iterrows())
    if max_plots is not None:
        plot_configs = plot_configs[:max_plots]

    save_dir_base = _get_save_path(data_name)

    for _, stat_row in plot_configs:
        current_method = stat_row[method_col]
        current_cfg = stat_row['cfg']
        
        current_labels, _ = GRIDS[current_method]['func'](X_processed, **current_cfg)
        
        member_data = []
        for i in range(X_processed.shape[0]):
            member_data.append({'id': ids[i], 'data': X_processed[i], 'label': current_labels[i]})

        
        member_data.sort(key=lambda x: (x['label'] == -1, x['label'], str(x['id'])))

        heatmap_plot_rows = []
        heatmap_yticklabels = []
        last_processed_label = None 

        if not member_data:
            print(f"No data to plot for heatmap: {current_method} cfg={current_cfg}")
            continue

        for item in member_data:
            
            if last_processed_label is not None and item['label'] != last_processed_label:
                if heatmap_yticklabels and heatmap_yticklabels[-1] != "": 
                    heatmap_plot_rows.append(np.full(X_processed.shape[1], np.nan))
                    heatmap_yticklabels.append("") # Separator row

            heatmap_plot_rows.append(item['data'])
            heatmap_yticklabels.append(str(item['id']))
            last_processed_label = item['label']
        
        if not heatmap_plot_rows:
            print(f"No rows to plot in heatmap for {current_method} cfg={current_cfg} after processing.")
            continue

        heatmap_array = np.array(heatmap_plot_rows)
        
        if heatmap_array.ndim == 1: 
             if X_processed.shape[1] > 0:
                heatmap_array = heatmap_array.reshape(-1, X_processed.shape[1])
             else: 
                continue
        
        if heatmap_array.shape[0] == 0 or heatmap_array.shape[1] == 0:
             print(f"Skipping heatmap for {current_method} cfg={current_cfg}: Resulting array is empty or has no features.")
             continue

        plt.figure(figsize=(10, max(6, len(heatmap_yticklabels) * 0.12))) 
        
        cmap = plt.get_cmap('viridis').copy()
        cmap.set_bad('white') 

        sns.heatmap(
            heatmap_array,
            yticklabels=heatmap_yticklabels,
            xticklabels=feature_names,
            cmap=cmap,
            cbar_kws={'label': 'Value'}, 
            mask=np.isnan(heatmap_array) 
        )
        
        num_clusters = len(set(l for l in current_labels if l != -1))
        silhouette_score_str = f"{stat_row.get('sil', 'N/A'):.3f}" if isinstance(stat_row.get('sil'), float) else "N/A"
        plt.title(f"Clustered Heatmap: {current_method}\ncfg: {current_cfg}\nClusters: {num_clusters}, Silhouette: {silhouette_score_str}")
        
        plt.xticks(ha='right', fontsize=7)
        
        ytick_fontsize = max(5, min(6, int(180 / len(heatmap_yticklabels))) if len(heatmap_yticklabels) > 30 else 9)
        plt.yticks(fontsize=ytick_fontsize)
        
        plt.tight_layout()

        param_str = '_'.join(f"{k}{v}" for k, v in current_cfg.items())
        fname = f"heatmap_sns_{current_method}_{param_str}.png"
        plt.savefig(save_dir_base / fname, dpi=150)
        plt.close()

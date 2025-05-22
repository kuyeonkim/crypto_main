from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Master save directory placeholder
SAVE_ROOT = Path("plots")  # customize later


# ----------------------------------------------------------------------
# Scatter visualization
# ----------------------------------------------------------------------

def visualize_by_config(df: pd.DataFrame, labels, algo_name: str, data_type: str, save_root: Path | None = None):
    """Scatter plot of the first two columns colored by cluster labels."""
    if save_root is None:
        save_root = SAVE_ROOT
    save_dir = save_root / data_type
    save_dir.mkdir(parents=True, exist_ok=True)

    plot_df = df.iloc[:, :2].copy()
    plot_df['cluster'] = labels

    fig, ax = plt.subplots()
    sns.scatterplot(data=plot_df, x=plot_df.columns[0], y=plot_df.columns[1], hue='cluster', ax=ax)
    ax.set_title(f"{algo_name} on {data_type}")
    fig.savefig(save_dir / f"{algo_name}_scatter.png")
    plt.close(fig)


# ----------------------------------------------------------------------
# Heatmap visualization
# ----------------------------------------------------------------------

def heatmap_for_clusters(df: pd.DataFrame, labels, algo_name: str, data_type: str, save_root: Path | None = None):
    """Create a heatmap of cluster means."""
    if save_root is None:
        save_root = SAVE_ROOT
    save_dir = save_root / data_type
    save_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df['cluster'] = labels
    mean_df = df.groupby('cluster').mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(mean_df, cmap='viridis', ax=ax)
    ax.set_title(f"{algo_name} cluster heatmap - {data_type}")
    fig.tight_layout()
    fig.savefig(save_dir / f"{algo_name}_heatmap.png")
    plt.close(fig)

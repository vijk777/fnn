import torch
import pandas as pd
import matplotlib.pyplot as plt


def load_metrics_csv(path_csv):
    """
    Load the training metrics CSV into a DataFrame.

    Parameters
    ----------
    path_csv : str or Path
        Full path to training_metrics.csv

    Returns
    -------
    pd.DataFrame
        DataFrame with at least an 'epoch' column and one or more metric columns.
    """
    df = pd.read_csv(path_csv)
    if 'epoch' not in df.columns:
        df.insert(0, 'epoch', range(len(df)))
    return df


def load_metrics_pt(path_pt):
    """
    Load the raw metrics .pt file.

    Parameters
    ----------
    path_pt : str or Path
        Full path to training_metrics.pt

    Returns
    -------
    dict
        {'epochs': [...], 'metrics': [{<metric>: value, …}, …]}
    """
    data = torch.load(path_pt, map_location='cpu')
    return data


def summarize_metrics(df, metric, mode='min'):
    """
    Find the epoch with the best value for a given metric.

    Parameters
    ----------
    df : pd.DataFrame
        As returned by load_metrics_csv.
    metric : str
        Column name to optimize.
    mode : {'min', 'max'}
        Whether "best" is the minimum (e.g. loss) or maximum (e.g. accuracy).

    Returns
    -------
    dict
        {'epoch': int, metric: float}
    """
    if mode == 'min':
        idx = df[metric].idxmin()
    else:
        idx = df[metric].idxmax()
    row = df.iloc[idx]
    return {'epoch': int(row['epoch']), metric: float(row[metric])}


def plot_metrics(df, metrics, figsize=(8, 4)):
    """
    Plot one or more metrics vs epoch.

    Parameters
    ----------
    df : pd.DataFrame
        As returned by load_metrics_csv.
    metrics : list of str
        Column names in df to plot.
    figsize : tuple
        Figure size, default (8,4).
    """
    fig, ax = plt.subplots(figsize=figsize)
    for m in metrics:
        if m not in df.columns:
            raise ValueError(f"Metric '{m}' not in DataFrame")
        ax.plot(df['epoch'], df[m], label=m)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Value')
    ax.set_title('Training Metrics')
    ax.legend()
    return fig, ax
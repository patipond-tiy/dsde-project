import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_by_bins(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bins: List[float],
    labels: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model performance stratified by target value ranges.

    This function bins the actual values and computes MAE, RMSE, and R2
    for each bin separately, allowing analysis of model performance across
    different ranges (e.g., quick vs slow resolutions).

    Parameters
    ----------
    y_true : np.ndarray
        True target values (actual resolution times)
    y_pred : np.ndarray
        Predicted target values
    bins : List[float]
        Bin edges for categorizing target values (e.g., [0, 7, 30, 90, 180, 365])
    labels : List[str]
        Bin labels (e.g., ['0-7d', '7-30d', '30-90d', '90-180d', '180-365d'])

    Returns
    -------
    Dict[str, Dict[str, float]]
        Nested dictionary with metrics for each bin:
        {
            '0-7d': {'mae': 1.5, 'rmse': 2.3, 'r2': 0.85, 'count': 10000, 'mean_true': 3.2, 'mean_pred': 3.0},
            '7-30d': {...},
            ...
        }

    Example
    -------
    >>> bins = [0, 7, 30, 90, 180, 365]
    >>> labels = ['0-7d', '7-30d', '30-90d', '90-180d', '180-365d']
    >>> metrics = evaluate_by_bins(y_test, y_pred, bins, labels)
    >>> print(f"High-value MAE: {metrics['180-365d']['mae']:.2f}")
    """

    # Create DataFrame for easier grouping
    eval_df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred
    })

    # Bin the true values
    eval_df['bin'] = pd.cut(eval_df['y_true'], bins=bins, labels=labels, include_lowest=True)

    results = {}

    for bin_label in labels:
        bin_mask = eval_df['bin'] == bin_label
        bin_data = eval_df[bin_mask]

        if len(bin_data) == 0:
            results[bin_label] = {
                'mae': np.nan,
                'rmse': np.nan,
                'r2': np.nan,
                'count': 0,
                'mean_true': np.nan,
                'mean_pred': np.nan
            }
            continue

        y_true_bin = bin_data['y_true'].values
        y_pred_bin = bin_data['y_pred'].values

        mae = mean_absolute_error(y_true_bin, y_pred_bin)
        rmse = np.sqrt(mean_squared_error(y_true_bin, y_pred_bin))

        # R2 can fail if variance is 0
        try:
            r2 = r2_score(y_true_bin, y_pred_bin)
        except:
            r2 = np.nan

        results[bin_label] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'count': len(bin_data),
            'mean_true': y_true_bin.mean(),
            'mean_pred': y_pred_bin.mean()
        }

    return results


def plot_predictions_by_range(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bins: List[float],
    labels: List[str],
    save_path: str,
    title: str = "Predictions by Target Range",
    max_points: int = 2000
):
    """
    Create a color-coded scatter plot showing predictions by target value range.

    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted target values
    bins : List[float]
        Bin edges for categorizing target values
    labels : List[str]
        Bin labels
    save_path : str
        Path to save the plot image
    title : str
        Plot title
    max_points : int
        Maximum number of points to plot (for performance)

    Returns
    -------
    None
        Saves plot to save_path
    """

    # Convert to numpy arrays if needed
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)

    # Sample points if too many
    if len(y_true) > max_points:
        indices = np.random.choice(len(y_true), size=max_points, replace=False)
        y_true_plot = y_true[indices]
        y_pred_plot = y_pred[indices]
    else:
        y_true_plot = y_true
        y_pred_plot = y_pred

    # Create DataFrame
    plot_df = pd.DataFrame({
        'y_true': y_true_plot,
        'y_pred': y_pred_plot
    })
    plot_df['bin'] = pd.cut(plot_df['y_true'], bins=bins, labels=labels, include_lowest=True)

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left plot: Scatter with color by bin
    palette = sns.color_palette("husl", len(labels))
    for i, bin_label in enumerate(labels):
        bin_data = plot_df[plot_df['bin'] == bin_label]
        if len(bin_data) > 0:
            axes[0].scatter(
                bin_data['y_true'],
                bin_data['y_pred'],
                alpha=0.4,
                s=20,
                color=palette[i],
                label=f"{bin_label} (n={len(bin_data)})"
            )

    # Add diagonal line (perfect predictions)
    max_val = max(y_true_plot.max(), y_pred_plot.max())
    axes[0].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect')
    axes[0].set_xlabel('Actual (days)', fontsize=12)
    axes[0].set_ylabel('Predicted (days)', fontsize=12)
    axes[0].set_title(title, fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper left', fontsize=9)
    axes[0].grid(alpha=0.3)

    # Right plot: Residuals by bin
    plot_df['residual'] = plot_df['y_pred'] - plot_df['y_true']

    # Box plot of residuals
    bin_order = labels
    sns.boxplot(
        data=plot_df,
        x='bin',
        y='residual',
        order=bin_order,
        palette=palette,
        ax=axes[1]
    )
    axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Target Range', fontsize=12)
    axes[1].set_ylabel('Residual (Predicted - Actual)', fontsize=12)
    axes[1].set_title('Residuals by Range', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3, axis='y')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved plot to: {save_path}")


def calculate_balance_score(metrics: Dict[str, Dict[str, float]]) -> float:
    """
    Calculate a balance score measuring how evenly the model performs across ranges.

    Lower score = more balanced performance
    Score is the coefficient of variation (CV) of MAEs across bins.

    Parameters
    ----------
    metrics : Dict[str, Dict[str, float]]
        Output from evaluate_by_bins()

    Returns
    -------
    float
        Balance score (lower is better, 0 = perfect balance)
    """

    maes = [m['mae'] for m in metrics.values() if not np.isnan(m['mae'])]

    if len(maes) == 0:
        return np.nan

    # Coefficient of Variation: std / mean
    # Lower CV = more consistent performance across ranges
    cv = np.std(maes) / np.mean(maes)

    return cv

import pandas as pd
import numpy as np
from typing import Dict, List

def hybrid_resample_regression(
    df: pd.DataFrame,
    target_col: str,
    bins: List[float],
    target_distribution: Dict[str, int],
    random_state: int = 42
) -> pd.DataFrame:
    """
    Apply hybrid resampling (undersample + oversample) for imbalanced regression.

    This function addresses class imbalance in regression problems by:
    1. Binning the continuous target variable
    2. Undersampling over-represented bins
    3. Oversampling under-represented bins with noise injection

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with features and target
    target_col : str
        Name of the target column (e.g., 'target_capped')
    bins : List[float]
        Bin edges for categorizing target values (e.g., [0, 7, 30, 90, 180, 365])
    target_distribution : Dict[str, int]
        Desired sample count for each bin (e.g., {'0-7d': 60000, '7-30d': 50000, ...})
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        Resampled dataframe with balanced target distribution and 'bin' column added

    Example
    -------
    >>> bins = [0, 7, 30, 90, 180, 365]
    >>> target_dist = {'0-7d': 60000, '7-30d': 50000, '30-90d': 40000,
    ...                '90-180d': 30000, '180-365d': 20000}
    >>> df_balanced = hybrid_resample_regression(df, 'target_capped', bins, target_dist)
    """

    np.random.seed(random_state)

    # Create bin labels
    labels = list(target_distribution.keys())

    # Bin the target column
    df = df.copy()
    df['bin'] = pd.cut(df[target_col], bins=bins, labels=labels, include_lowest=True)

    # Identify numerical vs categorical columns (for noise injection)
    # We'll add noise only to numerical features when oversampling
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove target and bin columns from numerical features
    numerical_cols = [col for col in numerical_cols if col not in [target_col, 'bin']]

    resampled_dfs = []

    print(f"\n=== Resampling Strategy ===")

    for bin_label in labels:
        bin_df = df[df['bin'] == bin_label].copy()
        current_count = len(bin_df)
        target_count = target_distribution[bin_label]

        if current_count == 0:
            print(f"{bin_label}: No samples - skipping")
            continue

        if current_count > target_count:
            # UNDERSAMPLE: Randomly select subset
            sampled_df = bin_df.sample(n=target_count, random_state=random_state)
            pct_change = ((target_count - current_count) / current_count) * 100
            print(f"{bin_label}: {current_count:,} → {target_count:,} ({pct_change:+.1f}%) [UNDERSAMPLE]")
            resampled_dfs.append(sampled_df)

        elif current_count < target_count:
            # OVERSAMPLE: Duplicate samples and add noise
            n_duplicates = target_count - current_count

            # Randomly sample rows to duplicate (with replacement)
            duplicated_indices = np.random.choice(bin_df.index, size=n_duplicates, replace=True)
            duplicated_df = bin_df.loc[duplicated_indices].copy()

            # Add Gaussian noise to numerical features
            # Use 2% of feature std as noise level (very conservative to preserve signal)
            noise_level = 0.02

            for col in numerical_cols:
                if col in duplicated_df.columns:
                    feature_std = bin_df[col].std()
                    if feature_std > 0:  # Only add noise if feature has variance
                        noise = np.random.normal(0, noise_level * feature_std, size=len(duplicated_df))
                        duplicated_df[col] = duplicated_df[col] + noise

            # Combine original and duplicated samples
            combined_df = pd.concat([bin_df, duplicated_df], ignore_index=True)

            pct_change = ((target_count - current_count) / current_count) * 100
            print(f"{bin_label}: {current_count:,} → {target_count:,} ({pct_change:+.1f}%) [OVERSAMPLE + noise]")
            resampled_dfs.append(combined_df)

        else:
            # Exact match - no resampling needed
            print(f"{bin_label}: {current_count:,} → {target_count:,} (0.0%) [NO CHANGE]")
            resampled_dfs.append(bin_df)

    # Combine all resampled bins
    df_resampled = pd.concat(resampled_dfs, ignore_index=True)

    # Shuffle to mix bins
    df_resampled = df_resampled.sample(frac=1, random_state=random_state).reset_index(drop=True)

    print(f"\n✅ Resampling complete: {len(df):,} → {len(df_resampled):,} samples")
    print(f"Distribution: {df_resampled['bin'].value_counts().sort_index().to_dict()}\n")

    return df_resampled

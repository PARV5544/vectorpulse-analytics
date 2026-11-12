import io
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def parse_tabular_csv(raw_bytes: bytes) -> Dict[str, Any]:
    """Parse CSV bytes into numeric matrix summary and column statistics."""
    df = pd.read_csv(io.BytesIO(raw_bytes))
    if df.empty:
        raise ValueError("Uploaded CSV dataset is empty.")

    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    if numeric_df.empty:
        raise ValueError("Dataset must contain at least one non-null numeric column.")

    return {
        "rows": int(len(df)),
        "numeric_rows": int(len(numeric_df)),
        "columns": list(df.columns),
        "numeric_columns": list(numeric_df.columns),
        "summary": {
            col: {
                "mean": float(numeric_df[col].mean()),
                "std": float(numeric_df[col].std(ddof=0)),
                "min": float(numeric_df[col].min()),
                "max": float(numeric_df[col].max()),
            }
            for col in numeric_df.columns
        },
        "matrix": numeric_df.to_numpy(dtype=float).tolist(),
    }


def compute_pca(
    matrix: List[List[float]],
    n_components: int = 2,
    standardize: bool = True,
) -> Dict[str, Any]:
    """Execute SVD-based Principal Component Analysis on a 2D numeric matrix."""
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("Input matrix must be 2-dimensional.")

    n_samples, n_features = arr.shape
    if n_samples < 2 or n_features < 1:
        raise ValueError("Matrix must have at least 2 rows and 1 feature column.")

    max_components = min(n_samples, n_features)
    if n_components < 1 or n_components > max_components:
        raise ValueError(
            f"n_components must be between 1 and {max_components}, got {n_components}."
        )

    centered = arr - np.mean(arr, axis=0)
    if standardize:
        stds = np.std(centered, axis=0, ddof=0)
        stds = np.where(stds == 0.0, 1.0, stds)
        centered = centered / stds

    # SVD decomposition: X = U * S * Vt
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    projected = u[:, :n_components] * s[:n_components]

    explained_variance = (s**2) / (n_samples - 1)
    total_variance = np.sum(explained_variance)
    if total_variance > 0:
        explained_variance_ratio = explained_variance / total_variance
    else:
        explained_variance_ratio = np.zeros_like(explained_variance)

    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_components": n_components,
        "explained_variance_ratio": [
            round(float(v), 6) for v in explained_variance_ratio[:n_components]
        ],
        "cumulative_variance_ratio": round(
            float(np.sum(explained_variance_ratio[:n_components])), 6
        ),
        "components": np.round(vt[:n_components], 6).tolist(),
        "projected": np.round(projected, 6).tolist(),
    }


def compute_matrix_op(matrix: List[List[float]], operation: str) -> Dict[str, Any]:
    """Run vectorized matrix transformations (normalize, covariance, correlation)."""
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 1:
        raise ValueError("Matrix must be 2D with at least 2 rows and 1 column.")

    if operation == "normalize":
        means = np.mean(arr, axis=0)
        stds = np.std(arr, axis=0, ddof=0)
        stds = np.where(stds == 0.0, 1.0, stds)
        result = (arr - means) / stds
    elif operation == "covariance":
        result = np.cov(arr, rowvar=False)
        if result.ndim == 0:
            result = np.array([[float(result)]])
    elif operation == "correlation":
        result = np.corrcoef(arr, rowvar=False)
        if result.ndim == 0:
            result = np.array([[1.0]])
        result = np.nan_to_num(result, nan=0.0)
    else:
        raise ValueError(f"Unsupported matrix operation: {operation}")

    return {
        "operation": operation,
        "shape": list(result.shape),
        "result": np.round(result, 6).tolist(),
    }

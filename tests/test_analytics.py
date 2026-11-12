import numpy as np
import pytest

from app.analytics import compute_matrix_op, compute_pca, parse_tabular_csv


def test_compute_pca_dimensionality_reduction():
    matrix = [
        [2.5, 2.4, 1.1],
        [0.5, 0.7, 0.3],
        [2.2, 2.9, 1.4],
        [1.9, 2.2, 1.0],
        [3.1, 3.0, 1.7],
        [2.3, 2.7, 1.2],
    ]
    res = compute_pca(matrix, n_components=2, standardize=True)
    assert res["n_samples"] == 6
    assert res["n_features"] == 3
    assert res["n_components"] == 2
    assert len(res["projected"]) == 6
    assert len(res["projected"][0]) == 2
    assert 0.9 <= res["cumulative_variance_ratio"] <= 1.0


def test_compute_pca_invalid_components():
    with pytest.raises(ValueError, match="n_components"):
        compute_pca([[1.0, 2.0], [3.0, 4.0]], n_components=5)


def test_compute_pca_zero_variance_column():
    matrix = [
        [1.0, 5.0],
        [2.0, 5.0],
        [3.0, 5.0],
    ]
    res = compute_pca(matrix, n_components=1, standardize=True)
    assert res["explained_variance_ratio"][0] == pytest.approx(1.0)


def test_compute_matrix_ops():
    matrix = [
        [1.0, 2.0],
        [2.0, 4.0],
        [3.0, 6.0],
    ]
    norm = compute_matrix_op(matrix, "normalize")
    assert norm["shape"] == [3, 2]
    assert np.allclose(np.mean(norm["result"], axis=0), [0.0, 0.0], atol=1e-5)

    cov = compute_matrix_op(matrix, "covariance")
    assert cov["shape"] == [2, 2]
    assert cov["result"][0][0] == pytest.approx(1.0)

    corr = compute_matrix_op(matrix, "correlation")
    assert corr["shape"] == [2, 2]
    assert corr["result"][0][1] == pytest.approx(1.0)


def test_compute_matrix_op_invalid():
    with pytest.raises(ValueError, match="Unsupported"):
        compute_matrix_op([[1.0, 2.0], [3.0, 4.0]], "eigen")


def test_parse_tabular_csv():
    csv_bytes = b"id,feature_a,feature_b,label\n1,10.0,20.0,ok\n2,20.0,40.0,ok\n"
    parsed = parse_tabular_csv(csv_bytes)
    assert parsed["rows"] == 2
    assert parsed["numeric_rows"] == 2
    assert "label" not in parsed["numeric_columns"]
    assert parsed["summary"]["feature_a"]["mean"] == pytest.approx(15.0)


def test_parse_tabular_csv_no_numeric():
    with pytest.raises(ValueError, match="numeric column"):
        parse_tabular_csv(b"a,b\nx,y\n")

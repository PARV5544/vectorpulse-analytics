from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["cache_backend"] in {"redis", "in-memory-fallback"}


def test_pca_endpoint_and_cache_hit_miss():
    payload = {
        "matrix": [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
        "n_components": 2,
        "standardize": True,
    }
    first = client.post("/api/v1/analytics/pca", json=payload)
    assert first.status_code == 200
    assert first.headers["X-Cache"] == "MISS"

    second = client.post("/api/v1/analytics/pca", json=payload)
    assert second.status_code == 200
    assert second.headers["X-Cache"] == "HIT"
    assert first.json() == second.json()


def test_matrix_ops_endpoint():
    payload = {
        "matrix": [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
        "operation": "correlation",
    }
    resp = client.post("/api/v1/analytics/matrix-ops", json=payload)
    assert resp.status_code == 200
    assert resp.json()["result"][0][1] == 1.0


def test_ingest_csv_endpoint():
    csv_content = b"x,y,tag\n1.0,2.0,a\n3.0,4.0,b\n5.0,6.0,c\n"
    resp = client.post(
        "/api/v1/datasets/ingest",
        files={"file": ("sample.csv", csv_content, "text/csv")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["filename"] == "sample.csv"
    assert data["numeric_rows"] == 3
    assert data["numeric_columns"] == ["x", "y"]


def test_ingest_empty_csv_returns_400():
    resp = client.post(
        "/api/v1/datasets/ingest",
        files={"file": ("empty.csv", b"", "text/csv")},
    )
    assert resp.status_code == 400

import asyncio
import time
from contextlib import asynccontextmanager
from typing import List, Literal

from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from pydantic import BaseModel, Field

from app.analytics import compute_matrix_op, compute_pca, parse_tabular_csv
from app.cache import AnalyticsCache
from app.config import settings

cache = AnalyticsCache(
    redis_url=settings.redis_url,
    ttl_seconds=settings.cache_ttl_seconds,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    await cache.close()


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="High-throughput tabular ingestion, PCA dimensionality reduction, and Redis-cached vector visualization endpoints.",
    lifespan=lifespan,
)


class PCARequest(BaseModel):
    matrix: List[List[float]] = Field(..., min_length=2)
    n_components: int = Field(default=2, ge=1)
    standardize: bool = True


class MatrixOpRequest(BaseModel):
    matrix: List[List[float]] = Field(..., min_length=2)
    operation: Literal["normalize", "covariance", "correlation"]


@app.get("/api/v1/health")
async def health_check():
    backend = await cache.backend_status()
    return {
        "status": "ok",
        "cache_backend": backend,
        "cache_hits": cache.hits,
        "cache_misses": cache.misses,
    }


@app.post("/api/v1/datasets/ingest")
async def ingest_csv(file: UploadFile = File(...)):
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    try:
        parsed = await asyncio.to_thread(parse_tabular_csv, raw)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "filename": file.filename,
        "rows": parsed["rows"],
        "numeric_rows": parsed["numeric_rows"],
        "columns": parsed["columns"],
        "numeric_columns": parsed["numeric_columns"],
        "summary": parsed["summary"],
    }


@app.post("/api/v1/analytics/pca")
async def run_pca(payload: PCARequest, response: Response):
    start = time.perf_counter()
    cache_key = cache.build_key("pca", payload.model_dump())
    cached = await cache.get(cache_key)
    if cached is not None:
        response.headers["X-Cache"] = "HIT"
        response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - start) * 1000:.2f}"
        return cached

    try:
        result = await asyncio.to_thread(
            compute_pca,
            payload.matrix,
            payload.n_components,
            payload.standardize,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    await cache.set(cache_key, result)
    response.headers["X-Cache"] = "MISS"
    response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - start) * 1000:.2f}"
    return result


@app.post("/api/v1/analytics/matrix-ops")
async def run_matrix_op(payload: MatrixOpRequest, response: Response):
    start = time.perf_counter()
    cache_key = cache.build_key("matrix-op", payload.model_dump())
    cached = await cache.get(cache_key)
    if cached is not None:
        response.headers["X-Cache"] = "HIT"
        response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - start) * 1000:.2f}"
        return cached

    try:
        result = await asyncio.to_thread(
            compute_matrix_op, payload.matrix, payload.operation
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    await cache.set(cache_key, result)
    response.headers["X-Cache"] = "MISS"
    response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - start) * 1000:.2f}"
    return result

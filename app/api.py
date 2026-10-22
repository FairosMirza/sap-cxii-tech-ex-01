"""HTTP service: GET /find_similar_products and GET /semantic_search, plus health and readiness probes.

Handlers are plain functions on purpose: the work is CPU-bound (NumPy, ONNX Runtime, FAISS), so FastAPI runs
them in its thread pool; `async def` would block the event loop during scoring.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Annotated, Callable, List, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from app.retrieval import FilterNotApplicableError, ProductNotFoundError, SimilarFilters, SimilarityEngine
from app.semantic import SemanticSearcher

logger = logging.getLogger("app.api")
MAX_RESULTS = 100
PRODUCT_ID = Annotated[str, Query(pattern=r"^[0-9a-f]{32}$", description="32-character uniq_id")]
COUNT = Annotated[int, Query(ge=1, le=MAX_RESULTS)]
MODE = Annotated[Literal["exact", "ann"] | None, Query(description="override the configured search mode")]
Loaded = tuple[SimilarityEngine, SemanticSearcher, dict]


def load_from_artifacts() -> Loaded:
    from app.artifacts import load_service

    return load_service()


def create_app(loader: Callable[[], Loaded] = load_from_artifacts) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = app.state.searcher = None
        app.state.manifest, app.state.not_ready_reason = {}, "starting"
        try:
            app.state.engine, app.state.searcher, app.state.manifest = loader()
            app.state.not_ready_reason = None
            logger.info("ready: %s", app.state.manifest.get("serving"))
        except Exception as exc:  # stay alive but unready, so the probe reports it and logs say why
            app.state.not_ready_reason = type(exc).__name__
            logger.error("artifacts could not be loaded: %s: %s", type(exc).__name__, exc)
        yield

    app = FastAPI(title="Product similarity service", version="1.0", lifespan=lifespan)

    @app.middleware("http")
    async def request_log(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("unhandled error, request %s %s %s", request_id, request.method, request.url.path)
            response = JSONResponse({"detail": "internal error", "request_id": request_id}, status_code=500)
        response.headers["x-request-id"] = request_id
        logger.info("%s %s %s %d %.1fms", request_id, request.method, request.url.path, response.status_code,
                    1000 * (time.perf_counter() - started))
        return response

    def ready_engine() -> tuple[SimilarityEngine, SemanticSearcher]:
        if app.state.engine is None:
            raise HTTPException(503, "service not ready")
        return app.state.engine, app.state.searcher

    @app.get("/find_similar_products", response_model=List[str])
    def find_similar_products(product_id: PRODUCT_ID, num_similar: COUNT, same_brand: bool = False,
                              min_price: Annotated[float | None, Query(ge=0)] = None,
                              max_price: Annotated[float | None, Query(ge=0)] = None, mode: MODE = None) -> List[str]:
        """IDs of the most similar products, most similar first. Exactly `num_similar` unless fewer are eligible.

        Optional filters (v2 filtered search): `same_brand`, `min_price`, `max_price` (rupees).
        """
        engine, _ = ready_engine()
        try:
            filters = SimilarFilters(same_brand, min_price, max_price)
            query = engine.row_of(product_id)
            return [engine.store.catalog.ids[row] for row, _ in engine.search(query, num_similar, filters=filters, mode=mode).hits]
        except ProductNotFoundError:
            raise HTTPException(404, f"product {product_id} not found") from None
        except (FilterNotApplicableError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from None
        except LookupError:
            raise HTTPException(422, "approximate search is not available for the configured text representation") from None

    @app.get("/semantic_search", response_model=List[str])
    def semantic_search(query: Annotated[str, Query(min_length=1, max_length=200)], num_results: COUNT,
                        mode: MODE = None) -> List[str]:
        """IDs of products matching a free-text query, filters (price, audience, colour) applied first.

        `num_results` is a maximum: fewer, or none, are returned when fewer products pass the filters.
        """
        _, searcher = ready_engine()
        try:
            result = searcher.search(query, num_results, mode=mode)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        return [searcher.store.catalog.ids[row] for row in result.rows]

    @app.get("/parse_query")
    def parse_query_endpoint(query: Annotated[str, Query(min_length=1, max_length=200)]) -> dict:
        """The constraints /semantic_search extracts from a query, for inspection."""
        from app.query_parser import parse_query

        try:
            parsed = parse_query(query)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        info = asdict(parsed)
        info["colors"], info["materials"] = sorted(parsed.colors), sorted(parsed.materials)
        return info

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> JSONResponse:
        if app.state.engine is None:
            return JSONResponse({"status": "not ready", "reason": app.state.not_ready_reason}, status_code=503)
        manifest = app.state.manifest
        return JSONResponse({"status": "ready", "dataset_sha256": manifest.get("dataset", {}).get("sha256"),
                             "serving": manifest.get("serving"), "code_commit": manifest.get("code_commit")})

    return app


app = create_app()

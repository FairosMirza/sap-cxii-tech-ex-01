# Product similarity service, two stages:
#   build    unpack the dataset and run `python -m scripts.prepare` (TF-IDF, ONNX embeddings, FAISS index,
#            pinned model files, manifest). Downloads the pinned model once, so it needs network access.
#   runtime  serving dependencies only (no PyTorch, no scikit-learn), non-root, health-checked.
# Build:  docker build --build-arg GIT_COMMIT=$(git rev-parse HEAD) -t product-similarity .
# Run:    docker run --rm -p 8000:8000 product-similarity
ARG PYTHON_IMAGE=python:3.12-slim

FROM ${PYTHON_IMAGE} AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /build
COPY requirements.txt requirements-build.txt ./
RUN pip install -r requirements-build.txt
COPY data/archive.zip data/
RUN python -m zipfile -e data/archive.zip data/ && rm data/archive.zip
COPY app app
COPY configs configs
COPY scripts scripts
ARG GIT_COMMIT=""
RUN GIT_COMMIT="${GIT_COMMIT}" python -m scripts.prepare --out artifacts

FROM ${PYTHON_IMAGE}
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /srv
COPY requirements.txt .
RUN pip install -r requirements.txt && useradd --system --uid 10001 --no-create-home service
COPY --from=build /build/data data
COPY --from=build /build/artifacts artifacts
COPY app app
COPY configs configs
USER service
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=4).status != 200)"]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

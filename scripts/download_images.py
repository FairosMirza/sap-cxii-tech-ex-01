"""Download each product's primary catalogue image: bounded, validated and resumable.

Run from the repository root:  python -m scripts.download_images
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import time
from collections import Counter
from pathlib import Path

import httpx
from PIL import Image

from app.data import DEFAULT_DATA_FILE, load_records
from app.features import IMAGE_HOSTS, primary_image_url

# Redirects are followed only within Amazon's image CDN.
ALLOWED_HOSTS = IMAGE_HOSTS | {"m.media-amazon.com"}
MAX_BYTES = 5 * 1024 * 1024
MAX_PIXELS = 40_000_000
MAX_SIDE = 320
ATTEMPTS = 3
TIMEOUT = httpx.Timeout(15.0, connect=5.0)
USER_AGENT = "sap-similarity-exercise-image-fetcher/1.0"


class _Retryable(Exception):
    pass


class _Rejected(Exception):
    pass


async def _check_host(request: httpx.Request) -> None:
    if request.url.scheme != "https" or request.url.host not in ALLOWED_HOSTS:
        raise _Rejected("disallowed_host")


def _decode_and_save(data: bytes, target: Path) -> None:
    with Image.open(io.BytesIO(data)) as image:
        if image.width * image.height > MAX_PIXELS:
            raise _Rejected("too_many_pixels")
        image.draft("RGB", (MAX_SIDE, MAX_SIDE))
        rgb = image.convert("RGB")
    rgb.thumbnail((MAX_SIDE, MAX_SIDE))
    tmp = target.with_suffix(".part")
    rgb.save(tmp, "JPEG", quality=88)
    tmp.replace(target)


async def _fetch(client: httpx.AsyncClient, url: str, target: Path) -> tuple[str, int]:
    status = "error"
    for attempt in range(1, ATTEMPTS + 1):
        try:
            async with client.stream("GET", url) as response:
                if response.status_code == 429 or response.status_code >= 500:
                    raise _Retryable(f"http_{response.status_code}")
                if response.status_code != 200:
                    return f"http_{response.status_code}", attempt
                if not response.headers.get("content-type", "").startswith("image/"):
                    return "not_image", attempt
                if int(response.headers.get("content-length") or 0) > MAX_BYTES:
                    return "too_large", attempt
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > MAX_BYTES:
                        return "too_large", attempt
            await asyncio.to_thread(_decode_and_save, bytes(body), target)
            return "ok", attempt
        except _Rejected as exc:
            return str(exc), attempt
        except httpx.TimeoutException:
            status = "timeout"
        except httpx.HTTPError:
            status = "network_error"
        except _Retryable as exc:
            status = str(exc)
        except (Image.UnidentifiedImageError, Image.DecompressionBombError, OSError):
            return "decode_error", attempt
        if attempt < ATTEMPTS:
            await asyncio.sleep(2 ** (attempt - 1))
    return status, ATTEMPTS


async def download_all(jobs: list[tuple[str, str | None]], out_dir: Path, concurrency: int) -> list[dict]:
    queue: asyncio.Queue = asyncio.Queue()
    for job in jobs:
        queue.put_nowait(job)
    results: list[dict] = []
    started = time.monotonic()

    async with httpx.AsyncClient(
        timeout=TIMEOUT,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
        limits=httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency),
        event_hooks={"request": [_check_host]},
    ) as client:

        async def worker() -> None:
            while True:
                try:
                    uid, url = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                target = out_dir / f"{uid}.jpg"
                if url is None:
                    status, attempts = "missing_url", 0
                elif target.exists():
                    status, attempts = "ok_cached", 0
                else:
                    status, attempts = await _fetch(client, url, target)
                results.append({"id": uid, "status": status, "attempts": attempts})
                if len(results) % 1000 == 0:
                    rate = len(results) / (time.monotonic() - started)
                    print(f"{len(results)}/{len(jobs)} done ({rate:.0f}/s)", flush=True)

        await asyncio.gather(*(worker() for _ in range(concurrency)))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_DATA_FILE))
    parser.add_argument("--out", default="data/images")
    parser.add_argument("--report", default="reports/image_download_summary.json")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="only the first N products (for smoke tests)")
    args = parser.parse_args()

    records, _ = load_records(args.data)
    jobs = [(r["uniq_id"], primary_image_url(r)) for r in records][: args.limit]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    results = asyncio.run(download_all(jobs, out_dir, args.concurrency))
    with (out_dir / "_status.jsonl").open("w") as fh:
        for row in sorted(results, key=lambda r: r["id"]):
            fh.write(json.dumps(row) + "\n")

    counts = Counter(r["status"] for r in results)
    usable = sum(1 for uid, _ in jobs if (out_dir / f"{uid}.jpg").exists())
    summary = {
        "products": len(jobs),
        "usable_images": usable,
        "status_counts": dict(counts.most_common()),
        "seconds": round(time.monotonic() - started, 1),
        "max_bytes": MAX_BYTES,
        "max_side_px": MAX_SIDE,
        "attempts_per_image": ATTEMPTS,
        "concurrency": args.concurrency,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

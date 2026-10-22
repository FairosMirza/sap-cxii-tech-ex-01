"""Local labelling page for blind relevance judgements (listens on 127.0.0.1 only).

Run from the repository root:  python -m scripts.label_server --split dev   then open http://127.0.0.1:8765
A supplemental pool is served the same way:  --split dev --supplement [NAME] --port 8766
Shows the query and one candidate at a time (image, title, brand, price). The producing system, scores and
extracted category are never shown. Keys 0-3 grade, B goes back, N focuses the note box. Labels are appended
to eval/labels.jsonl; re-grading appends a newer line and the latest one counts.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response

from app.data import load_records

LABELS = Path("eval/labels.jsonl")
IMAGES = Path("data/images")
POOLS = Path("eval/pools")
ID_PATTERN = re.compile(r"[0-9a-f]{32}")
MAX_NOTE = 500
TASKS = ("similar", "semantic")
QUESTIONS = {
    "similar": "Would a shopper looking at the left product plausibly buy the right one instead?",
    "semantic": "How well does the product match what the search asks for?",
}
RUBRICS = {
    "similar": {3: "Strong substitute: would happily buy this instead", 2: "Reasonable substitute: same type and use, some differences",
                1: "Weak alternative: related, but a stretch", 0: "Irrelevant: not an alternative"},
    "semantic": {3: "Exactly what was asked", 2: "Relevant, with a minor mismatch", 1: "Loosely related", 0: "Irrelevant"},
}
PLACEHOLDER = ('<svg xmlns="http://www.w3.org/2000/svg" width="320" height="320"><rect width="100%" height="100%" '
               'fill="#eee"/><text x="50%" y="50%" text-anchor="middle" fill="#888" font-family="sans-serif">no image</text></svg>')
STYLE = """body{font-family:system-ui,sans-serif;margin:24px;color:#222} .row{display:flex;gap:24px;align-items:flex-start}
.card{width:340px;border:1px solid #ddd;border-radius:8px;padding:12px} .card img{width:320px;height:320px;object-fit:contain;background:#fafafa}
.query{font-size:26px;width:340px;padding:12px} .meta{color:#555;margin-top:6px} button{display:block;width:420px;text-align:left;
font-size:15px;margin:6px 0;padding:8px} button.current{outline:3px solid #2a7} .bar{margin-bottom:12px;color:#555}"""


def read_labels(path: Path = LABELS) -> dict[tuple[str, str, str, str], dict]:
    labels: dict[tuple[str, str, str, str], dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                labels[(row["task"], row["split"], row["query"], row["candidate"])] = row
    return labels


def create_app(split: str, records: list[dict] | None = None, pools_dir: Path = POOLS,
               labels_path: Path = LABELS, images_dir: Path = IMAGES, supplement: str | None = None) -> FastAPI:
    records = records if records is not None else load_records()[0]
    products = {r["uniq_id"]: r for r in records}
    suffix = "" if supplement is None else "_supplement" + (f"_{supplement}" if supplement else "")
    pools = {task: json.loads((pools_dir / f"{task}_{split}{suffix}.json").read_text())["queries"]
             for task in TASKS if (pools_dir / f"{task}_{split}{suffix}.json").exists()}
    items = {task: [(qi, ci) for qi, q in enumerate(pool) for ci in range(len(q["candidates"]))] for task, pool in pools.items()}
    labels = read_labels(labels_path)
    token = secrets.token_urlsafe(16)
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    def key(task: str, qi: int, ci: int) -> tuple[str, str, str, str]:
        q = pools[task][qi]
        return task, split, q["query"], q["candidates"][ci]

    def done(task: str) -> int:
        return sum(key(task, qi, ci) in labels for qi, ci in items[task])

    def card(pid: str, heading: str) -> str:
        r = products[pid]
        price = f"₹{r['sales_price']}" if r.get("sales_price") else "price unknown"
        brand = html.escape((r.get("brand") or "").strip() or "brand unknown")
        return (f'<div class="card"><b>{heading}</b><br><img src="/img/{pid}" alt=""><div>{html.escape(r.get("product_name") or "")}</div>'
                f'<div class="meta">{brand} · {price}</div></div>')

    def check(task: str, qi: int, ci: int) -> None:
        if task not in pools or not 0 <= qi < len(pools[task]) or not 0 <= ci < len(pools[task][qi]["candidates"]):
            raise HTTPException(404, "no such item")

    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        rows = "".join(f'<li><a href="/next/{t}">{t}</a>: {done(t)} / {len(items[t])} labelled</li>' for t in pools)
        return f"<html><head><style>{STYLE}</style></head><body><h2>Blind labelling ({split})</h2><ul>{rows}</ul></body></html>"

    @app.get("/next/{task}")
    def next_item(task: str) -> RedirectResponse:
        if task not in pools:
            raise HTTPException(404, "no such task")
        for qi, ci in items[task]:
            if key(task, qi, ci) not in labels:
                return RedirectResponse(f"/item/{task}/{qi}/{ci}", status_code=303)
        return RedirectResponse("/", status_code=303)

    @app.get("/item/{task}/{qi}/{ci}", response_class=HTMLResponse)
    def item(task: str, qi: int, ci: int) -> str:
        check(task, qi, ci)
        position = items[task].index((qi, ci))
        prev_qi, prev_ci = items[task][max(position - 1, 0)]
        q = pools[task][qi]
        current = labels.get(key(task, qi, ci), {})
        left = card(q["query"], "Query product") if task == "similar" else f'<div class="query">“{html.escape(q["query"])}”</div>'
        buttons = "".join(
            f'<button name="grade" value="{g}" class="{"current" if current.get("grade") == g else ""}">{g} · {html.escape(text)}</button>'
            for g, text in RUBRICS[task].items())
        note = html.escape(current.get("note", ""))
        return f"""<html><head><meta charset="utf-8"><title>{task} {position + 1}</title><style>{STYLE}</style></head><body>
<div class="bar">{task} · query {qi + 1}/{len(pools[task])} · candidate {ci + 1}/{len(q["candidates"])} · labelled {done(task)}/{len(items[task])} · <a href="/">home</a></div>
<p><b>{QUESTIONS[task]}</b></p>
<div class="row">{left}{card(q["candidates"][ci], "Candidate")}
<form method="post" action="/label"><input type="hidden" name="task" value="{task}"><input type="hidden" name="qi" value="{qi}">
<input type="hidden" name="ci" value="{ci}"><input type="hidden" name="token" value="{token}">{buttons}
<input id="note" name="note" maxlength="{MAX_NOTE}" size="52" placeholder="Optional note for uncertain cases (N)" value="{note}">
<p><a href="/item/{task}/{prev_qi}/{prev_ci}">← back (B)</a></p></form></div>
<script>document.addEventListener("keydown", e => {{
  if (document.activeElement.id === "note") {{ return; }}
  if (["0","1","2","3"].includes(e.key)) {{ document.querySelector('button[value="' + e.key + '"]').click(); }}
  else if (e.key === "b" || e.key === "B") {{ location.href = "/item/{task}/{prev_qi}/{prev_ci}"; }}
  else if (e.key === "n" || e.key === "N") {{ e.preventDefault(); document.getElementById("note").focus(); }}
}});</script></body></html>"""

    @app.post("/label")
    def label(task: str = Form(...), qi: int = Form(...), ci: int = Form(...), grade: int = Form(...),
              token_: str = Form(..., alias="token"), note: str = Form("")) -> RedirectResponse:
        if not secrets.compare_digest(token_, token):
            raise HTTPException(403, "invalid session token")
        check(task, qi, ci)
        if grade not in RUBRICS[task] or len(note) > MAX_NOTE:
            raise HTTPException(422, "grade must be 0-3 and the note at most 500 characters")
        k = key(task, qi, ci)
        row = {"task": task, "split": split, "query": k[2], "candidate": k[3], "grade": grade, "note": note.strip(),
               "labelled_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        with labels_path.open("a") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        labels[k] = row
        return RedirectResponse(f"/next/{task}", status_code=303)

    @app.get("/img/{pid}")
    def image(pid: str) -> Response:
        if not ID_PATTERN.fullmatch(pid):
            raise HTTPException(404, "no such image")
        path = images_dir / f"{pid}.jpg"
        return FileResponse(path) if path.exists() else Response(PLACEHOLDER, media_type="image/svg+xml")

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "holdout"], default="dev")
    parser.add_argument("--supplement", nargs="?", const="", default=None, metavar="NAME",
                        help="serve a supplemental pool of the split (unnamed, or the given name)")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    uvicorn.run(create_app(args.split, supplement=args.supplement), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

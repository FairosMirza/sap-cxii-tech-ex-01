"""Loading and validating the raw LDJSON catalogue."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import orjson

DEFAULT_DATA_FILE = Path(
    "data/marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson"
)
MAX_REJECTED_FRACTION = 0.01
_ID_PATTERN = re.compile(r"[0-9a-f]{32}")


class DataQualityError(ValueError):
    """Raised when too much of the catalogue is unusable to continue safely."""


@dataclass
class LoadReport:
    path: str
    sha256: str
    rows: int = 0
    malformed_lines: list[int] = field(default_factory=list)
    invalid_id_lines: list[int] = field(default_factory=list)
    duplicate_id_lines: list[int] = field(default_factory=list)

    @property
    def rejected(self) -> int:
        return len(self.malformed_lines) + len(self.invalid_id_lines) + len(self.duplicate_id_lines)


def load_records(path: str | Path = DEFAULT_DATA_FILE) -> tuple[list[dict], LoadReport]:
    """Parse one JSON object per line, rejecting malformed rows and invalid or repeated IDs.

    Rejected rows are reported by line number; loading fails if more than 1% are rejected.
    """
    raw = Path(path).read_bytes()
    report = LoadReport(path=str(path), sha256=hashlib.sha256(raw).hexdigest())
    records: list[dict] = []
    seen: set[str] = set()
    for line_no, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = orjson.loads(line)
        except orjson.JSONDecodeError:
            report.malformed_lines.append(line_no)
            continue
        uid = record.get("uniq_id") if isinstance(record, dict) else None
        if not isinstance(uid, str) or not _ID_PATTERN.fullmatch(uid):
            report.invalid_id_lines.append(line_no)
            continue
        if uid in seen:
            report.duplicate_id_lines.append(line_no)
            continue
        seen.add(uid)
        records.append(record)

    report.rows = len(records)
    total = report.rows + report.rejected
    if total == 0 or report.rejected > MAX_REJECTED_FRACTION * total:
        raise DataQualityError(
            f"{report.rejected} of {total} lines rejected "
            f"(malformed={report.malformed_lines[:5]}, invalid_id={report.invalid_id_lines[:5]}, "
            f"duplicate_id={report.duplicate_id_lines[:5]})"
        )
    return records, report

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Iterable, Iterator


def expand_inputs(items: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for item in items:
        p = Path(item)
        if p.is_dir():
            files.extend(sorted(x for x in p.rglob("*") if x.suffix.lower() in {".txt", ".jsonl"}))
        else:
            matches = [Path(x) for x in glob.glob(item, recursive=True)]
            files.extend(matches if matches else [p])
    out = []
    seen = set()
    for p in files:
        rp = p.resolve()
        if rp.exists() and rp not in seen:
            seen.add(rp)
            out.append(rp)
    return out


def iter_documents(items: Iterable[str], text_field: str = "text") -> Iterator[str]:
    for path in expand_inputs(items):
        suffix = path.suffix.lower()
        if suffix == ".txt":
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    text = line.strip()
                    if text:
                        yield text
        elif suffix == ".jsonl":
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    text = obj.get(text_field, "")
                    if isinstance(text, str) and text.strip():
                        yield text.strip()
        else:
            raise ValueError(f"unsupported file: {path}")

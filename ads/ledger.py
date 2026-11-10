"""The evidence ledger: an append-only record of every fact the tools computed."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from ads.state import Evidence


def to_jsonable(obj: Any) -> Any:
    """Convert numpy/pandas/Path objects into plain JSON-safe Python."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, pd.Series):
        return to_jsonable(obj.to_dict())
    if isinstance(obj, pd.DataFrame):
        return to_jsonable(obj.to_dict(orient="records"))
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


class Ledger:
    """Append-only. Evidence is created ONLY through `add()` (called by the tool runner)."""

    def __init__(self) -> None:
        self._items: List[Evidence] = []
        self._index: Dict[str, Evidence] = {}
        self._cache: Dict[Tuple[str, str, str], str] = {}

    # ---- container protocol -------------------------------------------------
    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Evidence]:
        return iter(self._items)

    def __contains__(self, ev_id: object) -> bool:
        return ev_id in self._index

    # ---- writing ------------------------------------------------------------
    @staticmethod
    def _key(tool: str, args: Dict[str, Any], dataset_version: str) -> Tuple[str, str, str]:
        return (tool, json.dumps(to_jsonable(args or {}), sort_keys=True), dataset_version)

    def add(
        self,
        tool: str,
        args: Dict[str, Any],
        result: Any,
        summary: str,
        created_by: str = "unknown",
        dataset_version: str = "v0",
        step: int = 0,
        cacheable: bool = True,
    ) -> Evidence:
        ev = Evidence(
            id=f"E{len(self._items) + 1:04d}",
            tool=tool,
            args=to_jsonable(args or {}),
            result=to_jsonable(result),
            summary=summary,
            created_by=created_by,
            dataset_version=dataset_version,
            step=step,
        )
        self._items.append(ev)
        self._index[ev.id] = ev
        if cacheable:
            self._cache[self._key(tool, args, dataset_version)] = ev.id
        return ev

    # ---- reading ------------------------------------------------------------
    def get(self, ev_id: str) -> Evidence:
        return self._index[ev_id]

    def all(self) -> List[Evidence]:
        return list(self._items)

    def find(self, tool: Optional[str] = None, created_by: Optional[str] = None) -> List[Evidence]:
        return [
            e
            for e in self._items
            if (tool is None or e.tool == tool) and (created_by is None or e.created_by == created_by)
        ]

    def find_cached(self, tool: str, args: Dict[str, Any], dataset_version: str) -> Optional[Evidence]:
        ev_id = self._cache.get(self._key(tool, args, dataset_version))
        return self._index[ev_id] if ev_id else None

    def digest(self, max_items: Optional[int] = None) -> str:
        """Compact text an LLM can read: one line per fact."""
        items = self._items if max_items is None else self._items[-max_items:]
        return "\n".join(f"{e.id} [{e.tool}] {e.summary}" for e in items)

    # ---- persistence --------------------------------------------------------
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([e.to_dict() for e in self._items], indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Ledger":
        ledger = cls()
        for d in json.loads(Path(path).read_text(encoding="utf-8")):
            ev = Evidence.from_dict(d)
            ledger._items.append(ev)
            ledger._index[ev.id] = ev
        return ledger

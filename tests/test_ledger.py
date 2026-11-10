import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from ads.ledger import Ledger, to_jsonable


def test_ids_are_sequential_and_entries_immutable():
    ledger = Ledger()
    a = ledger.add("t1", {"x": 1}, {"v": 1}, "first")
    b = ledger.add("t2", {}, {"v": 2}, "second")
    assert (a.id, b.id) == ("E0001", "E0002")
    assert len(ledger) == 2 and "E0002" in ledger
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.summary = "changed"  # Evidence is a frozen dataclass


def test_cache_key_depends_on_tool_args_and_dataset_version():
    ledger = Ledger()
    ev = ledger.add("t", {"a": 1, "b": 2}, 1, "s", dataset_version="v0")
    assert ledger.find_cached("t", {"b": 2, "a": 1}, "v0").id == ev.id  # arg order is irrelevant
    assert ledger.find_cached("t", {"a": 1, "b": 2}, "v1") is None      # different data version
    assert ledger.find_cached("t", {"a": 9, "b": 2}, "v0") is None      # different args


def test_non_cacheable_entries_are_never_returned_from_cache():
    ledger = Ledger()
    ledger.add("clean", {}, 1, "s", cacheable=False)
    assert ledger.find_cached("clean", {}, "v0") is None


def test_to_jsonable_handles_numpy_and_nan():
    obj = {"i": np.int64(3), "f": np.float64(0.5), "nan": float("nan"), "arr": np.array([1, 2]),
           "s": pd.Series({"a": 1.0}), "b": np.bool_(True)}
    out = to_jsonable(obj)
    json.dumps(out)  # must be serialisable
    assert out["i"] == 3 and out["nan"] is None and out["arr"] == [1, 2] and out["b"] is True


def test_save_and_load_roundtrip(tmp_path):
    ledger = Ledger()
    ledger.add("t", {"a": 1}, {"rate": 0.427}, "Rate is 42.7%.", created_by="eda_agent", step=3)
    path = tmp_path / "ledger.json"
    ledger.save(path)
    loaded = Ledger.load(path)
    assert len(loaded) == 1
    ev = loaded.get("E0001")
    assert ev.result == {"rate": 0.427} and ev.created_by == "eda_agent" and ev.step == 3


def test_digest_is_one_line_per_fact():
    ledger = Ledger()
    ledger.add("t", {}, 1, "fact one")
    ledger.add("u", {}, 2, "fact two")
    assert ledger.digest().splitlines() == ["E0001 [t] fact one", "E0002 [u] fact two"]

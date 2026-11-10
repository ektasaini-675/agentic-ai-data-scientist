"""Importing this package registers every tool."""
from ads.tools.registry import execute_tool, get_tool, list_tools, register, tool_schemas  # noqa: F401
from ads.tools import data as _data  # noqa: F401  (profile, clean, identify_target)
from ads.tools import stats as _stats  # noqa: F401  (test_association, segment_rates, top_correlations)
from ads.tools import ml as _ml  # noqa: F401  (train_cv, permutation_importance, detect_leakage)
from ads.tools import viz as _viz  # noqa: F401  (plots)

__all__ = ["execute_tool", "get_tool", "list_tools", "register", "tool_schemas"]

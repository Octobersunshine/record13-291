from __future__ import annotations

from typing import Any

import pandas as pd


SUPPORTED_OPS = frozenset({
    "eq", "neq",
    "gt", "gte", "lt", "lte",
    "in", "not_in",
    "contains", "not_contains",
    "starts_with", "ends_with",
    "between", "not_between",
    "is_null", "is_not_null",
    "regex",
})


class FilterConditionError(Exception):
    pass


def _validate_condition(cond: dict[str, Any]) -> None:
    if "logic" in cond:
        if cond["logic"] not in ("AND", "OR"):
            raise FilterConditionError(
                f"logic must be 'AND' or 'OR', got '{cond['logic']}'"
            )
        if "conditions" not in cond or not isinstance(cond["conditions"], list):
            raise FilterConditionError(
                "logical group must contain a 'conditions' list"
            )
        for sub in cond["conditions"]:
            _validate_condition(sub)
        return

    for key in ("field", "op"):
        if key not in cond:
            raise FilterConditionError(f"condition missing required key: '{key}'")

    op = cond["op"]
    if op not in SUPPORTED_OPS:
        raise FilterConditionError(
            f"unsupported operator '{op}', supported: {sorted(SUPPORTED_OPS)}"
        )

    no_value_ops = {"is_null", "is_not_null"}
    if op not in no_value_ops and "value" not in cond:
        raise FilterConditionError(
            f"operator '{op}' requires a 'value' field"
        )

    if op in ("between", "not_between"):
        val = cond["value"]
        if not isinstance(val, (list, tuple)) or len(val) != 2:
            raise FilterConditionError(
                f"operator '{op}' requires 'value' to be a list/tuple of exactly 2 elements"
            )

    if op in ("in", "not_in"):
        val = cond["value"]
        if not isinstance(val, (list, tuple, set)):
            raise FilterConditionError(
                f"operator '{op}' requires 'value' to be a list/tuple/set"
            )


def _build_mask(df: pd.DataFrame, cond: dict[str, Any]) -> pd.Series:
    if "logic" in cond:
        logic = cond["logic"]
        sub_conditions = cond["conditions"]
        if not sub_conditions:
            return pd.Series(True, index=df.index)

        masks = [_build_mask(df, sub) for sub in sub_conditions]

        if logic == "AND":
            result = masks[0]
            for m in masks[1:]:
                result = result & m
            return result
        else:
            result = masks[0]
            for m in masks[1:]:
                result = result | m
            return result

    field = cond["field"]
    op = cond["op"]

    if field not in df.columns:
        raise FilterConditionError(f"field '{field}' not found in DataFrame columns")

    series = df[field]
    val = cond.get("value")

    if op == "eq":
        if pd.isna(val):
            return series.isna()
        return (series == val).fillna(False)
    if op == "neq":
        if pd.isna(val):
            return series.notna()
        return (series != val).fillna(False)
    if op == "gt":
        return (series > val).fillna(False)
    if op == "gte":
        return (series >= val).fillna(False)
    if op == "lt":
        return (series < val).fillna(False)
    if op == "lte":
        return (series <= val).fillna(False)
    if op == "in":
        return series.isin(val).fillna(False)
    if op == "not_in":
        return (~series.isin(val)).fillna(False)
    if op == "contains":
        return series.astype(str).str.contains(str(val), na=False)
    if op == "not_contains":
        return ~series.astype(str).str.contains(str(val), na=False)
    if op == "starts_with":
        return series.astype(str).str.startswith(str(val), na=False)
    if op == "ends_with":
        return series.astype(str).str.endswith(str(val), na=False)
    if op == "between":
        lo, hi = val
        return series.between(lo, hi).fillna(False)
    if op == "not_between":
        lo, hi = val
        return (~series.between(lo, hi)).fillna(False)
    if op == "is_null":
        return series.isna()
    if op == "is_not_null":
        return series.notna()
    if op == "regex":
        return series.astype(str).str.contains(str(val), regex=True, na=False)

    raise FilterConditionError(f"unhandled operator '{op}'")


class FilterService:
    def __init__(self, df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        self._df = df

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._df

    def filter(
        self,
        condition: dict[str, Any],
        *,
        validate: bool = True,
    ) -> pd.DataFrame:
        if validate:
            _validate_condition(condition)

        mask = _build_mask(self._df, condition)
        return self._df.loc[mask].reset_index(drop=True)

    def count(
        self,
        condition: dict[str, Any],
        *,
        validate: bool = True,
    ) -> int:
        if validate:
            _validate_condition(condition)

        mask = _build_mask(self._df, condition)
        return int(mask.sum())

    def mask(
        self,
        condition: dict[str, Any],
        *,
        validate: bool = True,
    ) -> pd.Series:
        if validate:
            _validate_condition(condition)

        return _build_mask(self._df, condition)

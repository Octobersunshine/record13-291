from __future__ import annotations

import json
import os
from datetime import datetime
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


class FilterTemplateError(Exception):
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


class FilterTemplateManager:
    def __init__(self, storage_path: str | None = None) -> None:
        if storage_path is None:
            storage_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "filter_templates.json",
            )
        self._storage_path = storage_path
        self._templates: dict[str, dict[str, Any]] = {}
        self._load()

    @property
    def storage_path(self) -> str:
        return self._storage_path

    def _load(self) -> None:
        if not os.path.exists(self._storage_path):
            self._templates = {}
            return
        try:
            with open(self._storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._templates = data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError) as e:
            raise FilterTemplateError(f"failed to load templates: {e}")

    def _save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self._storage_path), exist_ok=True)
            with open(self._storage_path, "w", encoding="utf-8") as f:
                json.dump(self._templates, f, ensure_ascii=False, indent=2)
        except OSError as e:
            raise FilterTemplateError(f"failed to save templates: {e}")

    def save(
        self,
        name: str,
        condition: dict[str, Any],
        *,
        description: str = "",
        validate: bool = True,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        if not isinstance(name, str) or not name.strip():
            raise FilterTemplateError("template name must be a non-empty string")

        name = name.strip()

        if name in self._templates and not overwrite:
            raise FilterTemplateError(
                f"template '{name}' already exists, set overwrite=True to replace"
            )

        if validate:
            _validate_condition(condition)

        now = datetime.now().isoformat(timespec="seconds")
        template = {
            "name": name,
            "description": description,
            "condition": condition,
            "created_at": self._templates.get(name, {}).get("created_at", now),
            "updated_at": now,
        }
        self._templates[name] = template
        self._save()
        return self._to_public(template)

    def load(self, name: str, *, validate: bool = True) -> dict[str, Any]:
        if name not in self._templates:
            raise FilterTemplateError(f"template '{name}' not found")

        template = self._templates[name]
        condition = template["condition"]

        if validate:
            _validate_condition(condition)

        return self._to_public(template)

    def load_condition(self, name: str, *, validate: bool = True) -> dict[str, Any]:
        return self.load(name, validate=validate)["condition"]

    def delete(self, name: str) -> bool:
        if name not in self._templates:
            return False
        del self._templates[name]
        self._save()
        return True

    def list_templates(self) -> list[dict[str, Any]]:
        return [self._to_public(t) for t in self._templates.values()]

    def exists(self, name: str) -> bool:
        return name in self._templates

    def rename(self, old_name: str, new_name: str) -> dict[str, Any]:
        if old_name not in self._templates:
            raise FilterTemplateError(f"template '{old_name}' not found")
        if not isinstance(new_name, str) or not new_name.strip():
            raise FilterTemplateError("new template name must be a non-empty string")
        new_name = new_name.strip()
        if new_name in self._templates:
            raise FilterTemplateError(f"template '{new_name}' already exists")

        template = self._templates.pop(old_name)
        template["name"] = new_name
        template["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self._templates[new_name] = template
        self._save()
        return self._to_public(template)

    def clear(self) -> None:
        self._templates = {}
        self._save()

    @staticmethod
    def _to_public(template: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": template["name"],
            "description": template.get("description", ""),
            "condition": template["condition"],
            "created_at": template.get("created_at"),
            "updated_at": template.get("updated_at"),
        }


class FilterService:
    def __init__(
        self,
        df: pd.DataFrame,
        template_manager: FilterTemplateManager | None = None,
    ) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        self._df = df
        self._template_manager = template_manager or FilterTemplateManager()

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._df

    @property
    def templates(self) -> FilterTemplateManager:
        return self._template_manager

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

    def filter_by_template(
        self,
        template_name: str,
        *,
        validate: bool = True,
    ) -> pd.DataFrame:
        condition = self._template_manager.load_condition(
            template_name, validate=validate
        )
        return self.filter(condition, validate=False)

    def count_by_template(
        self,
        template_name: str,
        *,
        validate: bool = True,
    ) -> int:
        condition = self._template_manager.load_condition(
            template_name, validate=validate
        )
        return self.count(condition, validate=False)

    def save_template(
        self,
        name: str,
        condition: dict[str, Any],
        *,
        description: str = "",
        validate: bool = True,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        return self._template_manager.save(
            name,
            condition,
            description=description,
            validate=validate,
            overwrite=overwrite,
        )

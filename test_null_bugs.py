import pandas as pd
import numpy as np
from filter_service import FilterService


def test_null_bugs():
    df = pd.DataFrame({
        "name": ["张三", "李四", "王五", None, "钱七", None],
        "age": [25, None, 28, 42, None, 55],
        "city": ["北京", "上海", None, "广州", "上海", "深圳"],
        "salary": [15000, np.nan, 18000, np.nan, 12000, 38000],
    })

    print("=== 原始数据 ===")
    print(df)
    print()

    svc = FilterService(df)

    # Bug 1: eq 比较 None 值时结果不正确
    cond1 = {"field": "name", "op": "eq", "value": None}
    result1 = svc.filter(cond1)
    print(f"=== Bug1: name eq None (期望 2 行) ===")
    print(result1)
    print(f"实际行数: {len(result1)}")
    print()

    # Bug 2: neq 比较 None 值时结果不正确
    cond2 = {"field": "name", "op": "neq", "value": None}
    result2 = svc.filter(cond2)
    print(f"=== Bug2: name neq None (期望 4 行) ===")
    print(result2)
    print(f"实际行数: {len(result2)}")
    print()

    # Bug 3: is_null 与 AND 组合时可能漏行
    cond3 = {
        "logic": "AND",
        "conditions": [
            {"field": "city", "op": "is_not_null", "value": None},
            {"field": "salary", "op": "gt", "value": 10000},
        ],
    }
    result3 = svc.filter(cond3)
    print(f"=== Bug3: city is_not_null AND salary > 10000 (期望 4 行) ===")
    print(result3)
    print(f"实际行数: {len(result3)}")
    print()

    # Bug 4: OR 组合时 mask 中有 NaN
    cond4 = {
        "logic": "OR",
        "conditions": [
            {"field": "age", "op": "gt", "value": 30},
            {"field": "name", "op": "eq", "value": "张三"},
        ],
    }
    mask4 = svc.mask(cond4)
    print(f"=== Bug4: mask 中是否有 NaN ===")
    print(f"mask: {mask4.tolist()}")
    print(f"mask 中 NaN 数量: {mask4.isna().sum()}")
    print()


if __name__ == "__main__":
    test_null_bugs()

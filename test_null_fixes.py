import pandas as pd
import numpy as np
from filter_service import FilterService


def test_null_fixes():
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

    all_passed = True

    # Test 1: eq None
    cond1 = {"field": "name", "op": "eq", "value": None}
    result1 = svc.filter(cond1)
    assert len(result1) == 2, f"Test1 failed: expected 2, got {len(result1)}"
    print("✓ Test1 passed: eq None")

    # Test 2: neq None
    cond2 = {"field": "name", "op": "neq", "value": None}
    result2 = svc.filter(cond2)
    assert len(result2) == 4, f"Test2 failed: expected 4, got {len(result2)}"
    print("✓ Test2 passed: neq None")

    # Test 3: eq np.nan
    cond3 = {"field": "salary", "op": "eq", "value": np.nan}
    result3 = svc.filter(cond3)
    assert len(result3) == 2, f"Test3 failed: expected 2, got {len(result3)}"
    print("✓ Test3 passed: eq np.nan")

    # Test 4: neq np.nan
    cond4 = {"field": "salary", "op": "neq", "value": np.nan}
    result4 = svc.filter(cond4)
    assert len(result4) == 4, f"Test4 failed: expected 4, got {len(result4)}"
    print("✓ Test4 passed: neq np.nan")

    # Test 5: is_null
    cond5 = {"field": "age", "op": "is_null"}
    result5 = svc.filter(cond5)
    assert len(result5) == 2, f"Test5 failed: expected 2, got {len(result5)}"
    print("✓ Test5 passed: is_null")

    # Test 6: is_not_null
    cond6 = {"field": "age", "op": "is_not_null"}
    result6 = svc.filter(cond6)
    assert len(result6) == 4, f"Test6 failed: expected 4, got {len(result6)}"
    print("✓ Test6 passed: is_not_null")

    # Test 7: gt with NaN in data (NaN > 30 应为 False)
    cond7 = {"field": "age", "op": "gt", "value": 30}
    result7 = svc.filter(cond7)
    assert len(result7) == 2, f"Test7 failed: expected 2, got {len(result7)}"  # 42, 55
    print("✓ Test7 passed: gt with NaN")

    # Test 8: lt with NaN in data (NaN < 30 应为 False)
    cond8 = {"field": "age", "op": "lt", "value": 30}
    result8 = svc.filter(cond8)
    assert len(result8) == 2, f"Test8 failed: expected 2, got {len(result8)}"  # 25, 28
    print("✓ Test8 passed: lt with NaN")

    # Test 9: AND with is_not_null
    cond9 = {
        "logic": "AND",
        "conditions": [
            {"field": "city", "op": "is_not_null"},
            {"field": "salary", "op": "gt", "value": 10000},
        ],
    }
    result9 = svc.filter(cond9)
    # 行 0 (北京,15000), 行 4 (上海,12000), 行 5 (深圳,38000)
    assert len(result9) == 3, f"Test9 failed: expected 3, got {len(result9)}"
    print("✓ Test9 passed: AND with is_not_null and gt")

    # Test 10: OR with is_null
    cond10 = {
        "logic": "OR",
        "conditions": [
            {"field": "name", "op": "is_null"},
            {"field": "salary", "op": "gt", "value": 30000},
        ],
    }
    result10 = svc.filter(cond10)
    # 行 3 (name None), 行 5 (name None, salary 38000)
    assert len(result10) == 2, f"Test10 failed: expected 2, got {len(result10)}"
    print("✓ Test10 passed: OR with is_null")

    # Test 11: between with NaN
    cond11 = {"field": "salary", "op": "between", "value": [12000, 20000]}
    result11 = svc.filter(cond11)
    # 15000, 18000, 12000
    assert len(result11) == 3, f"Test11 failed: expected 3, got {len(result11)}"
    print("✓ Test11 passed: between with NaN")

    # Test 12: in with NaN (NaN in list)
    cond12 = {"field": "name", "op": "in", "value": ["张三", "李四", None]}
    result12 = svc.filter(cond12)
    # 张三, 李四, None, None
    assert len(result12) == 4, f"Test12 failed: expected 4, got {len(result12)}"
    print("✓ Test12 passed: in with None in list")

    # Test 13: 嵌套组合: (name is_null OR age > 40) AND city is_not_null
    cond13 = {
        "logic": "AND",
        "conditions": [
            {
                "logic": "OR",
                "conditions": [
                    {"field": "name", "op": "is_null"},
                    {"field": "age", "op": "gt", "value": 40},
                ],
            },
            {"field": "city", "op": "is_not_null"},
        ],
    }
    result13 = svc.filter(cond13)
    # 行 3 (name None, age 42, city 广州), 行 5 (name None, age 55, city 深圳)
    assert len(result13) == 2, f"Test13 failed: expected 2, got {len(result13)}"
    print("✓ Test13 passed: nested with is_null")

    # Test 14: mask 中没有 NaN
    cond14 = {"field": "age", "op": "gt", "value": 100}
    mask14 = svc.mask(cond14)
    assert mask14.isna().sum() == 0, "Test14 failed: mask contains NaN"
    print("✓ Test14 passed: mask has no NaN")

    # Test 15: eq NaN 和 is_null 等价
    cond15a = {"field": "age", "op": "eq", "value": np.nan}
    cond15b = {"field": "age", "op": "is_null"}
    result15a = svc.filter(cond15a)
    result15b = svc.filter(cond15b)
    assert result15a.equals(result15b), "Test15 failed: eq NaN != is_null"
    print("✓ Test15 passed: eq NaN equals is_null")

    print()
    print("✓✓✓ All 15 tests passed! ✓✓✓")


if __name__ == "__main__":
    test_null_fixes()

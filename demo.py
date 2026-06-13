import pandas as pd

from filter_service import FilterService, FilterConditionError


def main():
    df = pd.DataFrame({
        "name": ["张三", "李四", "王五", "赵六", "钱七", "孙八", "周九", "吴十"],
        "age": [25, 35, 28, 42, 19, 55, 31, 22],
        "city": ["北京", "上海", "北京", "广州", "上海", "深圳", "北京", "杭州"],
        "salary": [15000, 25000, 18000, 32000, 12000, 38000, 20000, 14000],
        "department": ["技术", "销售", "技术", "管理", "销售", "管理", "技术", "人事"],
    })

    svc = FilterService(df)

    # 1) simple AND
    cond_and = {
        "logic": "AND",
        "conditions": [
            {"field": "age", "op": "gte", "value": 25},
            {"field": "city", "op": "eq", "value": "北京"},
        ],
    }
    print("=== AND: age>=25 AND city==北京 ===")
    print(svc.filter(cond_and))
    print()

    # 2) simple OR
    cond_or = {
        "logic": "OR",
        "conditions": [
            {"field": "city", "op": "eq", "value": "北京"},
            {"field": "city", "op": "eq", "value": "上海"},
        ],
    }
    print("=== OR: city==北京 OR city==上海 ===")
    print(svc.filter(cond_or))
    print()

    # 3) nested: (age>30 OR salary>30000) AND department==技术
    cond_nested = {
        "logic": "AND",
        "conditions": [
            {
                "logic": "OR",
                "conditions": [
                    {"field": "age", "op": "gt", "value": 30},
                    {"field": "salary", "op": "gt", "value": 30000},
                ],
            },
            {"field": "department", "op": "eq", "value": "技术"},
        ],
    }
    print("=== 嵌套: (age>30 OR salary>30000) AND department==技术 ===")
    print(svc.filter(cond_nested))
    print()

    # 4) in / not_in / between / contains
    cond_ops = {
        "logic": "AND",
        "conditions": [
            {"field": "department", "op": "in", "value": ["技术", "管理"]},
            {"field": "salary", "op": "between", "value": [15000, 25000]},
            {"field": "name", "op": "contains", "value": "张"},
        ],
    }
    print("=== IN + BETWEEN + CONTAINS ===")
    print(svc.filter(cond_ops))
    print()

    # 5) is_null / is_not_null
    df_with_null = df.copy()
    df_with_null.loc[1, "salary"] = None
    df_with_null.loc[3, "salary"] = None
    svc_null = FilterService(df_with_null)
    print("=== IS NULL: salary 为空 ===")
    print(svc_null.filter({"field": "salary", "op": "is_null", "value": None}))
    print()

    # 6) count
    print(f"=== COUNT: city==北京 → {svc.count({'field': 'city', 'op': 'eq', 'value': '北京'})} ===")
    print()

    # 7) validation error
    try:
        svc.filter({"field": "age", "op": "unknown_op", "value": 1})
    except FilterConditionError as e:
        print(f"=== 验证错误捕获: {e} ===")
    print()

    # 8) regex
    cond_regex = {
        "logic": "AND",
        "conditions": [
            {"field": "name", "op": "regex", "value": r"^[张王]"},
        ],
    }
    print("=== REGEX: name 以'张'或'王'开头 ===")
    print(svc.filter(cond_regex))


if __name__ == "__main__":
    main()

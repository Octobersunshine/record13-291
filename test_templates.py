import os
import tempfile

import pandas as pd

from filter_service import (
    FilterConditionError,
    FilterService,
    FilterTemplateManager,
    FilterTemplateError,
)


def test_template_manager():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = os.path.join(tmpdir, "templates.json")
        mgr = FilterTemplateManager(storage_path=storage)

        cond_tech_beijing = {
            "logic": "AND",
            "conditions": [
                {"field": "department", "op": "eq", "value": "技术"},
                {"field": "city", "op": "eq", "value": "北京"},
            ],
        }

        cond_high_salary = {
            "logic": "OR",
            "conditions": [
                {"field": "salary", "op": "gt", "value": 30000},
                {"field": "age", "op": "gte", "value": 40},
            ],
        }

        # Test 1: save
        t1 = mgr.save(
            "北京技术部",
            cond_tech_beijing,
            description="筛选北京地区技术部员工",
        )
        assert t1["name"] == "北京技术部"
        assert t1["description"] == "筛选北京地区技术部员工"
        assert t1["condition"] == cond_tech_beijing
        assert "created_at" in t1 and "updated_at" in t1
        print("✓ Test1 passed: save template")

        # Test 2: duplicate without overwrite
        try:
            mgr.save("北京技术部", cond_tech_beijing)
            assert False, "should have raised"
        except FilterTemplateError:
            pass
        print("✓ Test2 passed: duplicate save blocked")

        # Test 3: duplicate with overwrite
        t1b = mgr.save(
            "北京技术部",
            cond_tech_beijing,
            description="更新后的描述",
            overwrite=True,
        )
        assert t1b["description"] == "更新后的描述"
        assert t1b["created_at"] == t1["created_at"]
        assert t1b["updated_at"] >= t1["updated_at"]
        print("✓ Test3 passed: overwrite template preserves created_at")

        # Test 4: save second template
        t2 = mgr.save("高薪或资深", cond_high_salary)
        assert t2["name"] == "高薪或资深"
        print("✓ Test4 passed: save second template")

        # Test 5: list
        lst = mgr.list_templates()
        assert len(lst) == 2
        names = sorted(t["name"] for t in lst)
        assert names == ["北京技术部", "高薪或资深"]
        print("✓ Test5 passed: list templates")

        # Test 6: load
        loaded = mgr.load("北京技术部")
        assert loaded["condition"] == cond_tech_beijing
        print("✓ Test6 passed: load template")

        # Test 7: load_condition
        loaded_cond = mgr.load_condition("高薪或资深")
        assert loaded_cond == cond_high_salary
        print("✓ Test7 passed: load_condition")

        # Test 8: load not found
        try:
            mgr.load("不存在的模板")
            assert False, "should have raised"
        except FilterTemplateError:
            pass
        print("✓ Test8 passed: load not found raises")

        # Test 9: exists
        assert mgr.exists("北京技术部")
        assert not mgr.exists("不存在")
        print("✓ Test9 passed: exists")

        # Test 10: delete
        assert mgr.delete("高薪或资深")
        assert not mgr.exists("高薪或资深")
        assert len(mgr.list_templates()) == 1
        assert not mgr.delete("不存在")
        print("✓ Test10 passed: delete template")

        # Test 11: rename
        renamed = mgr.rename("北京技术部", "北京技术员工")
        assert renamed["name"] == "北京技术员工"
        assert not mgr.exists("北京技术部")
        assert mgr.exists("北京技术员工")
        print("✓ Test11 passed: rename template")

        # Test 12: persistence across instances
        mgr2 = FilterTemplateManager(storage_path=storage)
        assert mgr2.exists("北京技术员工")
        assert len(mgr2.list_templates()) == 1
        print("✓ Test12 passed: persistence across instances")

        # Test 13: clear
        mgr2.clear()
        assert len(mgr2.list_templates()) == 0
        mgr3 = FilterTemplateManager(storage_path=storage)
        assert len(mgr3.list_templates()) == 0
        print("✓ Test13 passed: clear templates")

        # Test 14: empty name
        try:
            mgr.save("", cond_tech_beijing)
            assert False
        except FilterTemplateError:
            pass
        try:
            mgr.save("   ", cond_tech_beijing)
            assert False
        except FilterTemplateError:
            pass
        print("✓ Test14 passed: empty name rejected")

        # Test 15: invalid condition rejected on save
        try:
            mgr.save("坏条件", {"field": "x", "op": "bad_op", "value": 1})
            assert False
        except FilterConditionError:
            pass
        print("✓ Test15 passed: invalid condition rejected on save")


def test_filter_service_with_templates():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = os.path.join(tmpdir, "templates.json")
        df = pd.DataFrame({
            "name": ["张三", "李四", "王五", "赵六", "钱七"],
            "age": [25, 35, 28, 42, 19],
            "city": ["北京", "上海", "北京", "广州", "上海"],
            "salary": [15000, 25000, 18000, 32000, 12000],
            "department": ["技术", "销售", "技术", "管理", "销售"],
        })
        svc = FilterService(df, template_manager=FilterTemplateManager(storage_path=storage))

        cond = {
            "logic": "AND",
            "conditions": [
                {"field": "department", "op": "eq", "value": "技术"},
                {"field": "city", "op": "eq", "value": "北京"},
            ],
        }

        # save via service
        svc.save_template("北京技术", cond, description="技术部北京员工")
        assert svc.templates.exists("北京技术")
        print("✓ Test F1 passed: save_template via FilterService")

        # filter_by_template
        result = svc.filter_by_template("北京技术")
        assert len(result) == 2
        assert set(result["name"].tolist()) == {"张三", "王五"}
        print("✓ Test F2 passed: filter_by_template")

        # count_by_template
        assert svc.count_by_template("北京技术") == 2
        print("✓ Test F3 passed: count_by_template")

        # default template_manager path
        svc2 = FilterService(df)
        assert svc2.templates is not None
        print("✓ Test F4 passed: default template_manager")


if __name__ == "__main__":
    test_template_manager()
    print()
    test_filter_service_with_templates()
    print()
    print("✓✓✓ All template tests passed! ✓✓✓")

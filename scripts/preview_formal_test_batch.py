import hashlib
import json
from collections import Counter

from tool_calling.data.templates import fill_placeholders
from tool_calling.data.validation import validate_date_range
from tool_calling.evaluation.validation import validate_tool_output
from tool_calling.paths import PROJECT_ROOT


PLAN_PATH = PROJECT_ROOT / "configs/final_evaluation_plan.json"
TEMPLATE_PATH = PROJECT_ROOT / "data/manual/formal/test_templates.json"
PARAMETER_PATH = (
    PROJECT_ROOT / "data/generated/formal/test_parameter_sets.json"
)
TOOL_PATH = PROJECT_ROOT / "configs/tools.json"
OUTPUT_PATH = PROJECT_ROOT / "data/generated/formal/test_candidates.jsonl"


def build_candidates():
    plan_bytes = PLAN_PATH.read_bytes()
    plan = json.loads(plan_bytes)

    # 必须先锁定最终方案，Test 数据不用于继续选择模型或调参。
    if (
        plan["status"] != "locked_before_test"
        or plan["test_policy"]["allow_test_tuning"] is not False
    ):
        raise ValueError("最终评估方案尚未锁定，或允许了 Test 调参。")

    template_bytes = TEMPLATE_PATH.read_bytes()
    parameter_bytes = PARAMETER_PATH.read_bytes()
    tool_bytes = TOOL_PATH.read_bytes()
    templates = json.loads(template_bytes)
    parameter_sets = json.loads(parameter_bytes)
    tools = json.loads(tool_bytes)
    tools_sha256 = hashlib.sha256(tool_bytes).hexdigest()

    if tools_sha256 != plan["frozen_inputs"]["tools_sha256"]:
        raise ValueError("工具定义与锁定方案不一致。")
    if not templates or not parameter_sets:
        raise ValueError("模板和参数组都不能为空。")

    for items, id_field in (
        (templates, "template_id"),
        (parameter_sets, "parameter_set_id"),
    ):
        ids = [item[id_field] for item in items]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{id_field} 存在重复。")
        if any(item["split"] != "test" for item in items):
            raise ValueError("正式 Test 来源中出现了其他分区。")

    tool_schemas = {
        tool["function"]["name"]: tool["function"]["parameters"]
        for tool in tools
    }
    category_names = {
        "electronics": "电子产品",
        "clothing": "服装",
        "home": "家居用品",
    }
    metric_names = {
        "revenue": "销售金额",
        "order_count": "订单数量",
        "units_sold": "售出件数",
    }
    # 来源指纹用于追溯数据版本，不属于模型的用户输入。
    source_info = {
        "stage": "formal",
        "templates_sha256": hashlib.sha256(template_bytes).hexdigest(),
        "parameter_sets_sha256": hashlib.sha256(parameter_bytes).hexdigest(),
        "tools_sha256": tools_sha256,
        "final_evaluation_plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
    }

    examples = []
    seen_targets = {}
    for parameter_set in parameter_sets:
        validate_date_range(parameter_set)
        threshold = parameter_set["threshold"]
        if type(threshold) is not int or threshold < 1:
            raise ValueError("正常参数组的库存阈值必须为正整数。")

        # 中文名称构造请求，英文枚举作为对应的工具参数。
        variables = {
            **parameter_set,
            "category_zh": category_names[parameter_set["category"]],
            "metric_zh": metric_names[parameter_set["metric"]],
        }
        for template in templates:
            query = fill_placeholders(template["query_template"], variables)
            expected = fill_placeholders(
                template["expected_template"], variables
            )
            errors = validate_tool_output(expected, tool_schemas)
            if errors:
                raise ValueError(f"{template['template_id']}：{errors}")

            target_key = json.dumps(
                expected, ensure_ascii=False, sort_keys=True
            )
            # 相同请求与目标保留第一条；矛盾目标直接拒绝。
            if query in seen_targets:
                if seen_targets[query] != target_key:
                    raise ValueError(f"同一请求存在不同参考目标：{query}")
                continue
            seen_targets[query] = target_key

            examples.append({
                "id": (
                    f"{template['template_id']}"
                    f"__{parameter_set['parameter_set_id']}"
                ),
                "template_id": template["template_id"],
                "family_id": template["family_id"],
                "parameter_set_id": parameter_set["parameter_set_id"],
                "split": "test",
                "case_type": template["case_type"],
                "query": query,
                "expected": expected,
                "source_info": source_info,
            })

    return examples, len(templates) * len(parameter_sets)


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"候选文件已存在，未覆盖：{OUTPUT_PATH}")

    # 全部候选构造并校验成功后再写入，此步尚未选取最终 200 条。
    examples, raw_count = build_candidates()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("x", encoding="utf-8") as output_file:
        for example in examples:
            output_file.write(
                json.dumps(example, ensure_ascii=False) + "\n"
            )

    print("原始组合数量：", raw_count)
    print("去除重复数量：", raw_count - len(examples))
    print("去重后候选数量：", len(examples))
    print("动作分布：", dict(Counter(
        example["expected"]["action"] for example in examples
    )))
    print("候选文件：", OUTPUT_PATH)


if __name__ == "__main__":
    main()
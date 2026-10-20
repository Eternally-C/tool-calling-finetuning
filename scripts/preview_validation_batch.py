import json
from collections import Counter
from pathlib import Path

from tool_calling.data.validation import validate_date_range
from tool_calling.data.templates import fill_placeholders
from tool_calling.evaluation.validation import validate_tool_output


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_json(relative_path):
    return json.loads(
        (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
    )


def main():
    templates = load_json("data/manual/validation_templates.json")
    parameter_sets = load_json(
        "data/manual/validation_parameter_sets.json"
    )
    train_templates = load_json("data/manual/train_templates.json")
    train_seeds = load_json("data/manual/train_examples.json")
    tools = load_json("configs/tools.json")

    if not templates or not parameter_sets:
        raise ValueError("Validation 模板或参数组为空。")

    # 来源编号必须唯一，才能准确关联生成样本与后续运行结果。
    for items, key in (
        (templates, "template_id"),
        (parameter_sets, "parameter_set_id"),
    ):
        values = [item[key] for item in items]
        if len(values) != len(set(values)):
            raise ValueError(f"来源字段 {key} 出现重复编号。")

    train_families = {
        template["family_id"] for template in train_templates
    }
    for template in templates:
        if template["split"] != "validation":
            raise ValueError("Validation 模板中出现其他分区。")
        if template["family_id"] in train_families:
            raise ValueError(
                f"表达家族与 Train 重叠：{template['family_id']}"
            )

    category_names = {
        "electronics": "电子产品",
        "clothing": "服装",
        "home": "家居用品",
    }
    tool_schemas = {
        tool["function"]["name"]: tool["function"]["parameters"]
        for tool in tools
    }

    # 读取已存在的 Train 请求，只用于检查跨分区重复。
    train_preview = [
        json.loads(line)
        for line in (
            PROJECT_ROOT / "data/generated/train_preview.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    train_queries = {
        sample["query"] for sample in train_preview + train_seeds
    }

    examples = []
    seen_query_targets = {}

    for parameter_set in parameter_sets:
        validate_date_range(parameter_set)

        for field in ("product_id", "order_id", "refund_id"):
            value = parameter_set[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"参数池中的 {field} 必须是非空字符串。")

        threshold = parameter_set["threshold"]
        if type(threshold) is not int or threshold < 1:
            raise ValueError("参数池中的库存阈值必须是正整数。")

        category = parameter_set["category"]
        if category not in category_names:
            raise ValueError("参数池中的商品分类不在允许列表中。")

        # 请求与参考目标共用变量，保留整数等参数的原始类型。
        variables = {
            **parameter_set,
            "category_zh": category_names[category],
        }

        for template in templates:
            query = fill_placeholders(
                template["query_template"], variables
            )
            expected = fill_placeholders(
                template["expected_template"], variables
            )

            if query in train_queries:
                raise ValueError(f"请求与 Train 重复：{query}")

            # 检查填充后的参考目标，避免不合法标签进入候选数据。
            errors = validate_tool_output(expected, tool_schemas)
            if errors:
                raise ValueError(
                    f"参考目标不合法：{template['template_id']}，{errors}"
                )

            target_key = json.dumps(
                expected, ensure_ascii=False, sort_keys=True
            )
            if query in seen_query_targets:
                if seen_query_targets[query] != target_key:
                    raise ValueError(f"同一请求存在不同参考目标：{query}")
                continue

            seen_query_targets[query] = target_key
            examples.append(
                {
                    "id": (
                        f"preview_{template['template_id']}"
                        f"__{parameter_set['parameter_set_id']}"
                    ),
                    "template_id": template["template_id"],
                    "family_id": template["family_id"],
                    "parameter_set_id": parameter_set["parameter_set_id"],
                    "split": template["split"],
                    "case_type": template["case_type"],
                    "query": query,
                    "expected": expected,
                }
            )

    sample_ids = [sample["id"] for sample in examples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("生成的样本编号重复。")

    # 保存可再生的预览文件；重复运行会更新该文件。
    output_path = PROJECT_ROOT / "data/generated/validation_preview.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for sample in examples:
            output_file.write(
                json.dumps(sample, ensure_ascii=False) + "\n"
            )

    raw_count = len(templates) * len(parameter_sets)
    print("候选样本文件：", output_path)
    print("原始组合数量：", raw_count)
    print("去除重复数量：", raw_count - len(examples))
    print("去重后样本数量：", len(examples))
    print(
        "动作分布：",
        dict(Counter(sample["expected"]["action"] for sample in examples)),
    )

    # 展示第一组参数对应的三类请求，完整记录保存在 JSONL 中。
    for sample in examples[:3]:
        print(json.dumps(sample, ensure_ascii=False))


if __name__ == "__main__":
    main()
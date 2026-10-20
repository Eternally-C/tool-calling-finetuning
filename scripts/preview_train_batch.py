import json
from collections import Counter
from pathlib import Path

from tool_calling.data.templates import fill_placeholders
from tool_calling.data.validation import validate_date_range


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    templates = json.loads(
        (PROJECT_ROOT / "data/manual/train_templates.json").read_text(
            encoding="utf-8"
        )
    )
    parameter_sets = json.loads(
        (PROJECT_ROOT / "data/manual/train_parameter_sets.json").read_text(
            encoding="utf-8"
        )
    )

    category_names = {
        "electronics": "电子产品",
        "clothing": "服装",
        "home": "家居用品",
    }

    examples = []
    seen_query_targets = {}

    # 外层遍历参数组，内层遍历模板，生成所有组合。
    for parameter_set in parameter_sets:
        validate_date_range(parameter_set)

        threshold = parameter_set["threshold"]
        if type(threshold) is not int or threshold < 1:
            raise ValueError("参数组中的库存阈值必须是正整数。")

        # 中文分类由英文枚举映射得到，保证请求与参考目标一致。
        variables = {
            **parameter_set,
            "category_zh": category_names[parameter_set["category"]],
        }

        for template in templates:
            if template["split"] != "train":
                raise ValueError("训练模板文件中出现了非 train 模板。")

            query = fill_placeholders(
                template["query_template"], variables
            )
            expected = fill_placeholders(
                template["expected_template"], variables
            )

            # 固定键顺序，避免仅因 JSON 字段排列不同而误判目标不同。
            target_key = json.dumps(
                expected, ensure_ascii=False, sort_keys=True
            )

            # 相同请求、相同目标保留第一条；相同请求、不同目标立即报错。
            if query in seen_query_targets:
                if seen_query_targets[query] != target_key:
                    raise ValueError(f"同一请求存在不同参考目标：{query}")
                continue

            seen_query_targets[query] = target_key
            examples.append(
                {
                    # 同一模板会生成多个样本，编号同时包含模板和参数组。
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

    raw_count = len(templates) * len(parameter_sets)
    action_counts = Counter(
        example["expected"]["action"] for example in examples
    )

    # 样本编号用于后续结果关联，保存前检查是否重复。
    sample_ids = [example["id"] for example in examples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("样本编号重复，请核对模板与参数组编号。")

    # 每行保存一条完整样本，保留来源字段，便于审查与错误追踪。
    output_path = PROJECT_ROOT / "data/generated/train_preview.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        for example in examples:
            output_file.write(
                json.dumps(example, ensure_ascii=False) + "\n"
            )

    print("候选样本文件：", output_path)
    print("模板数量：", len(templates))
    print("参数组数量：", len(parameter_sets))
    print("原始组合数量：", raw_count)
    print("去除重复数量：", raw_count - len(examples))
    print("去重后样本数量：", len(examples))
    print("动作分布：", dict(action_counts))

    # 展示各参数组的库存查询和阈值查询，便于核对替换值与参数类型。
    print("\n库存与阈值样例：")
    for example in examples:
        if example["template_id"] in {
            "inventory_direct_001",
            "low_stock_below_001",
        }:
            print(
                json.dumps(
                    {
                        "parameter_set_id": example["parameter_set_id"],
                        "query": example["query"],
                        "expected": example["expected"],
                    },
                    ensure_ascii=False,
                )
            )


if __name__ == "__main__":
    main()
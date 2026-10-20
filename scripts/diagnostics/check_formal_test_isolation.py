import json
from collections import Counter

from tool_calling.data.formal_isolation import check_formal_data_isolation
from tool_calling.data.isolation import (
    ID_FIELDS,
    collect_date_ranges,
    load_json,
)
from tool_calling.data.validation import validate_date_range
from tool_calling.evaluation.validation import validate_tool_output
from tool_calling.paths import PROJECT_ROOT


def check_formal_test_isolation():
    # 复用已有检查；训练数据准备仍只读取 Train / Validation。
    samples_by_split = check_formal_data_isolation()
    plan = load_json("configs/final_evaluation_plan.json")
    if (
        plan["status"] != "locked_before_test"
        or plan["test_policy"]["allow_test_tuning"] is not False
    ):
        raise ValueError("最终方案尚未锁定，或允许了 Test 调参。")

    parameters = load_json("data/generated/formal/test_parameter_sets.json")
    sample_path = PROJECT_ROOT / "data/generated/formal/test_selected.jsonl"
    samples = [
        json.loads(line)
        for line in sample_path.read_text(encoding="utf-8").splitlines()
    ]
    if not parameters or not samples:
        raise ValueError("Test 参数池或选样文件为空。")

    parameter_ids = {item["parameter_set_id"] for item in parameters}
    if len(parameter_ids) != len(parameters):
        raise ValueError("Test 参数组编号重复。")

    for parameter in parameters:
        if parameter["split"] != "test":
            raise ValueError("Test 参数池中出现其他分区。")
        validate_date_range(parameter)
        for field in ID_FIELDS:
            value = parameter[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Test 的 {field} 必须为非空字符串。")
        threshold = parameter["threshold"]
        if type(threshold) is not int or threshold < 1:
            raise ValueError("Test 正常参数池的阈值必须为正整数。")

    tools = load_json("configs/tools.json")
    tool_schemas = {
        tool["function"]["name"]: tool["function"]["parameters"]
        for tool in tools
    }
    for sample in samples:
        if sample["split"] != "test":
            raise ValueError(f"Test 选样分区错误：{sample['id']}")
        if sample["parameter_set_id"] not in parameter_ids:
            raise ValueError(f"找不到 Test 来源参数组：{sample['id']}")
        for field in ("id", "query", "template_id", "family_id"):
            value = sample[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Test 的 {field} 必须为非空字符串。")
        errors = validate_tool_output(sample["expected"], tool_schemas)
        if errors:
            raise ValueError(f"{sample['id']}：{errors}")

    for field in ("id", "query"):
        values = [sample[field] for sample in samples]
        if len(values) != len(set(values)):
            raise ValueError(f"Test 的 {field} 存在重复。")

    targets = plan["test_policy"]["target_action_counts"]
    data_targets = load_json("configs/formal_data_plan.json")[
        "target_action_counts"
    ]["test"]
    counts = Counter(sample["expected"]["action"] for sample in samples)
    if targets != data_targets or counts != targets:
        raise ValueError("Test 动作数量与锁定方案或数据计划不一致。")
    samples_by_split["test"] = samples
    print(f"test：{len(samples)} 条，动作分布：{dict(counts)}")

    records_by_split = {}
    for split, split_samples in samples_by_split.items():
        # 参数池包含请求中的干扰编号，合法调用补充实际工具参数。
        records_by_split[split] = load_json(
            f"data/generated/formal/{split}_parameter_sets.json"
        ) + [
            sample["expected"]["arguments"]
            for sample in split_samples
            if sample["expected"]["action"] == "call"
        ]

    for other in ("train", "validation"):
        for field in ("id", "query", "template_id", "family_id"):
            test_values = {sample[field] for sample in samples}
            other_values = {
                sample[field] for sample in samples_by_split[other]
            }
            if test_values & other_values:
                raise ValueError(f"{field} 在 Test 与 {other} 中重叠。")
            print(f"test / {other}，{field}：重叠 0")

        # 分类和统计指标是共享枚举，不要求互斥。
        for field in (*ID_FIELDS, "threshold"):
            test_values = {
                item[field] for item in records_by_split["test"]
                if field in item
            }
            other_values = {
                item[field] for item in records_by_split[other]
                if field in item
            }
            overlap = test_values & other_values
            if overlap:
                raise ValueError(
                    f"{field} 在 Test 与 {other} 中重叠：{sorted(overlap)}"
                )
            print(f"test / {other}，{field}：重叠 0")

        test_ranges = collect_date_ranges(records_by_split["test"])
        other_ranges = collect_date_ranges(records_by_split[other])
        # 包含首尾日期；共享一天也视为重叠，同日查询本身合法。
        for test_start, test_end in sorted(test_ranges):
            for other_start, other_end in sorted(other_ranges):
                if max(test_start, other_start) <= min(test_end, other_end):
                    raise ValueError(
                        f"有效日期区间在 Test 与 {other} 中重叠："
                        f"{test_start} 至 {test_end}；"
                        f"{other_start} 至 {other_end}"
                    )
        print(f"test / {other}，有效日期区间：重叠 0")

    print("正式 Train / Validation / Test 数据隔离检查通过。")
    return samples_by_split


if __name__ == "__main__":
    check_formal_test_isolation()
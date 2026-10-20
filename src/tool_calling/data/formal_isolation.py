import json
from collections import Counter

from tool_calling.data.isolation import (
    ID_FIELDS,
    collect_date_ranges,
    load_json,
)
from tool_calling.data.validation import validate_date_range
from tool_calling.evaluation.validation import validate_tool_output
from tool_calling.paths import PROJECT_ROOT


def check_formal_data_isolation():
    """检查正式 Train / Validation，返回通过检查的选样数据。"""
    tools = load_json("configs/tools.json")
    tool_schemas = {
        tool["function"]["name"]: tool["function"]["parameters"]
        for tool in tools
    }
    targets = load_json("configs/formal_data_plan.json")[
        "target_action_counts"
    ]
    samples_by_split = {}
    records_by_split = {}

    for split in ("train", "validation"):
        parameters = load_json(
            f"data/generated/formal/{split}_parameter_sets.json"
        )
        sample_path = (
            PROJECT_ROOT / f"data/generated/formal/{split}_selected.jsonl"
        )
        samples = [
            json.loads(line)
            for line in sample_path.read_text(encoding="utf-8").splitlines()
        ]
        if not parameters or not samples:
            raise ValueError(f"{split} 的参数池或选样文件为空。")

        parameter_ids = {
            item["parameter_set_id"] for item in parameters
        }
        if len(parameter_ids) != len(parameters):
            raise ValueError(f"{split} 的参数组编号重复。")

        # 正常参数必须有效；非法边界请求由对应模板单独构造。
        for parameter in parameters:
            if parameter["split"] != split:
                raise ValueError(f"{split} 参数池中出现其他分区。")
            validate_date_range(parameter)
            for field in ID_FIELDS:
                value = parameter[field]
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{split} 的 {field} 必须为非空字符串。")
            threshold = parameter["threshold"]
            if type(threshold) is not int or threshold < 1:
                raise ValueError(f"{split} 的正常阈值必须为正整数。")

        for sample in samples:
            if sample["split"] != split:
                raise ValueError(f"选样分区错误：{sample['id']}")
            if sample["parameter_set_id"] not in parameter_ids:
                raise ValueError(f"找不到来源参数组：{sample['id']}")
            if (
                not isinstance(sample["query"], str)
                or not sample["query"].strip()
            ):
                raise ValueError(f"请求为空：{sample['id']}")
            errors = validate_tool_output(sample["expected"], tool_schemas)
            if errors:
                raise ValueError(f"{sample['id']}：{errors}")

        for field in ("id", "query"):
            values = [sample[field] for sample in samples]
            if len(values) != len(set(values)):
                raise ValueError(f"{split} 的 {field} 存在重复。")

        counts = Counter(sample["expected"]["action"] for sample in samples)
        if counts != targets[split]:
            raise ValueError(f"{split} 的动作数量不符合正式数据计划。")

        samples_by_split[split] = samples
        # 参数池覆盖请求中的干扰编号，合法调用补充实际工具参数。
        records_by_split[split] = parameters + [
            sample["expected"]["arguments"]
            for sample in samples
            if sample["expected"]["action"] == "call"
        ]
        print(f"{split}：{len(samples)} 条，动作分布：{dict(counts)}")

    # 精确重复与已标记的表达族，都不能跨正式分区出现。
    for field in ("id", "query", "template_id", "family_id"):
        train_values = {
            sample[field] for sample in samples_by_split["train"]
        }
        validation_values = {
            sample[field] for sample in samples_by_split["validation"]
        }
        if train_values & validation_values:
            raise ValueError(f"{field} 在正式 Train / Validation 中重叠。")
        print(f"{field}：重叠 0")

    # 分类和指标是共享接口枚举，不要求跨分区互斥。
    for field in (*ID_FIELDS, "threshold"):
        train_values = {
            item[field] for item in records_by_split["train"]
            if field in item
        }
        validation_values = {
            item[field] for item in records_by_split["validation"]
            if field in item
        }
        overlap = train_values & validation_values
        if overlap:
            raise ValueError(f"{field} 跨正式分区重叠：{sorted(overlap)}")
        print(f"{field}：重叠 0")

    train_ranges = collect_date_ranges(records_by_split["train"])
    validation_ranges = collect_date_ranges(
        records_by_split["validation"]
    )
    # 区间包含首尾日期：共享一天也属于重叠；同日查询本身合法。
    for train_start, train_end in sorted(train_ranges):
        for validation_start, validation_end in sorted(validation_ranges):
            if max(train_start, validation_start) <= min(
                train_end, validation_end
            ):
                raise ValueError(
                    "正式分区有效日期区间重叠："
                    f"Train {train_start} 至 {train_end}；"
                    f"Validation {validation_start} 至 {validation_end}"
                )
    print("有效日期区间：重叠 0")
    print("正式 Train / Validation 数据隔离检查通过。")
    return samples_by_split


if __name__ == "__main__":
    check_formal_data_isolation()
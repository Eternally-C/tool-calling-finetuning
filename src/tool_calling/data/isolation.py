import json
from datetime import date

from tool_calling.data.validation import validate_date_range
from tool_calling.evaluation.validation import validate_tool_output
from tool_calling.paths import PROJECT_ROOT


ID_FIELDS = ("product_id", "order_id", "refund_id")


def load_json(relative_path):
    return json.loads(
        (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
    )


def collect_date_ranges(records):
    # 正常参数池和合法调用参数都已经过日期校验。
    return {
        (
            date.fromisoformat(record["start_date"]),
            date.fromisoformat(record["end_date"]),
        )
        for record in records
        if "start_date" in record and "end_date" in record
    }


def check_parameter_isolation():
    train_parameters = load_json(
        "data/manual/train_parameter_sets.json"
    )
    validation_parameters = load_json(
        "data/manual/validation_parameter_sets.json"
    )
    train_seeds = load_json("data/manual/train_examples.json")
    tools = load_json("configs/tools.json")
    tool_schemas = {
        tool["function"]["name"]: tool["function"]["parameters"]
        for tool in tools
    }

    # 正常参数池本身必须有效，非法参数案例由模板单独构造。
    for split, parameter_sets in (
        ("train", train_parameters),
        ("validation", validation_parameters),
    ):
        if not parameter_sets:
            raise ValueError(f"{split} 参数池为空。")

        for parameter_set in parameter_sets:
            validate_date_range(parameter_set)

            for field in ID_FIELDS:
                value = parameter_set[field]
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{split} 的 {field} 必须是非空字符串。")

            threshold = parameter_set["threshold"]
            if type(threshold) is not int or threshold < 1:
                raise ValueError(f"{split} 的阈值必须是正整数。")

    # 开发阶段已经使用的手写样例也属于 Train。
    # 先校验参考输出，再收集其中有效调用的参数。
    train_arguments = []
    for sample in train_seeds:
        expected = sample["expected"]
        errors = validate_tool_output(expected, tool_schemas)
        if errors:
            raise ValueError(
                f"Train 手写样例 {sample['id']} 的参考输出不合法：{errors}"
            )
        if expected["action"] == "call":
            train_arguments.append(expected["arguments"])

    train_records = train_parameters + train_arguments

    # 编号和正常阈值不能跨分区复用；固定枚举允许共享。
    for field in (*ID_FIELDS, "threshold"):
        train_values = {
            record[field] for record in train_records if field in record
        }
        validation_values = {
            record[field] for record in validation_parameters
        }
        overlap = train_values & validation_values

        if overlap:
            raise ValueError(
                f"{field} 在 Train 与 Validation 中重叠：{sorted(overlap)}"
            )
        print(f"{field}：重叠 0")

    train_ranges = collect_date_ranges(train_records)
    validation_ranges = collect_date_ranges(validation_parameters)

    # 日期范围包含首尾两天；即使起止日期不同，也要检查区间交集。
    for train_start, train_end in sorted(train_ranges):
        for validation_start, validation_end in sorted(validation_ranges):
            if max(train_start, validation_start) <= min(
                train_end, validation_end
            ):
                raise ValueError(
                    "有效日期区间重叠："
                    f"Train {train_start} 至 {train_end}；"
                    f"Validation {validation_start} 至 {validation_end}"
                )

    print("有效日期区间：重叠 0")
    print("Train / Validation 参数隔离检查通过。")


if __name__ == "__main__":
    check_parameter_isolation()
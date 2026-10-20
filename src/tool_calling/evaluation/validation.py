import json
from datetime import date

from tool_calling.paths import PROJECT_ROOT


def validate_argument(name, value, rule):
    """按工具定义检查单个参数的类型、长度、范围和枚举值。"""
    errors = []
    expected_type = rule.get("type")

    if expected_type == "string":
        if not isinstance(value, str):
            return [f"参数 {name} 必须是字符串。"]

        if len(value) < rule.get("minLength", 0):
            errors.append(f"参数 {name} 的字符串长度不足。")

        # 日期既要真实存在，也要符合项目约定的 YYYY-MM-DD 格式。
        if rule.get("format") == "date":
            try:
                parsed_date = date.fromisoformat(value)
            except ValueError:
                errors.append(
                    f"参数 {name} 必须是有效的 YYYY-MM-DD 日期。"
                )
            else:
                if parsed_date.isoformat() != value:
                    errors.append(
                        f"参数 {name} 必须使用 YYYY-MM-DD 格式。"
                    )

    elif expected_type == "integer":
        # 严格区分整数与布尔值，避免把 true 误当成库存阈值。
        if type(value) is not int:
            return [f"参数 {name} 必须是整数。"]

        if "minimum" in rule and value < rule["minimum"]:
            errors.append(
                f"参数 {name} 不能小于 {rule['minimum']}。"
            )

    else:
        # 新增工具使用其他类型时，需要先扩展检查能力。
        return [f"参数 {name} 使用了尚未支持的类型：{expected_type}"]

    if "enum" in rule and value not in rule["enum"]:
        errors.append(
            f"参数 {name} 必须属于允许列表：{rule['enum']}"
        )

    return errors


def validate_tool_output(output, tool_schemas):
    """检查输出结构、工具名称、参数名称与参数取值约束。"""
    errors = []

    if not isinstance(output, dict):
        return ["输出必须是 JSON 对象。"]

    action = output.get("action")
    if action not in ("call", "clarify", "unsupported"):
        return ["action 必须是 call、clarify 或 unsupported。"]

    # 根据输出自己的 action 检查结构，后续可直接用于模型预测。
    if action == "call":
        if set(output) != {"action", "name", "arguments"}:
            errors.append("call 只能包含 action、name、arguments 三个字段。")

        name = output.get("name")
        if not isinstance(name, str) or name not in tool_schemas:
            errors.append("工具名称缺失或不在当前工具列表中。")
            return errors

        arguments = output.get("arguments")
        if not isinstance(arguments, dict):
            errors.append("arguments 必须是 JSON 对象。")
            return errors

        schema = tool_schemas[name]
        required = set(schema.get("required", []))
        allowed = set(schema.get("properties", {}))

        # 从工具定义读取参数要求，避免另写一份名单造成不一致。
        missing = required - set(arguments)
        extra = set(arguments) - allowed

        if missing:
            errors.append(f"缺少必需参数：{sorted(missing)}")

        if schema.get("additionalProperties") is False and extra:
            errors.append(f"出现未定义的参数：{sorted(extra)}")

        # 使用同一份工具定义检查取值，未知参数已由上面的逻辑报告。
        for argument_name, value in arguments.items():
            if argument_name in allowed:
                errors.extend(
                    validate_argument(
                        argument_name,
                        value,
                        schema["properties"][argument_name],
                    )
                )

        # 先通过字段和单参数检查，再比较日期，避免无效输入触发异常。
        if name == "get_sales_summary" and not errors:
            start_date = date.fromisoformat(arguments["start_date"])
            end_date = date.fromisoformat(arguments["end_date"])

            # 工具允许查询同一天，只有结束日期更早时才拒绝。
            if end_date < start_date:
                errors.append("结束日期不能早于开始日期。")

    else:
        if set(output) != {"action", "message"}:
            errors.append("clarify 和 unsupported 只能包含 action、message。")

        message = output.get("message")
        if not isinstance(message, str) or not message.strip():
            errors.append("message 必须是非空字符串。")

    return errors


def main():
    tools = json.loads(
        (PROJECT_ROOT / "configs/tools.json").read_text(encoding="utf-8")
    )
    tool_schemas = {
        tool["function"]["name"]: tool["function"]["parameters"]
        for tool in tools
    }

    sample_path = PROJECT_ROOT / "data/generated/train_preview.jsonl"
    samples = [
        json.loads(line)
        for line in sample_path.read_text(encoding="utf-8").splitlines()
    ]

    if not samples:
        raise ValueError("候选样本文件为空。")

    failures = []

    # 本次检查 Train 候选样本的参考目标，保留编号便于定位问题。
    for sample in samples:
        if sample["split"] != "train":
            raise ValueError("Train 候选文件中出现了其他分区的样本。")

        errors = validate_tool_output(sample["expected"], tool_schemas)
        if errors:
            failures.append((sample["id"], errors))

    print("已检查参考目标：", len(samples))
    print("结构校验通过：", len(samples) - len(failures))
    print("结构校验未通过：", len(failures))

    for sample_id, errors in failures:
        print(sample_id, errors)

    if failures:
        raise ValueError("存在结构不合格的参考目标，请按样本编号检查。")


if __name__ == "__main__":
    main()
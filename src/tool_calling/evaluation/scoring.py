import argparse
import json

from tool_calling.evaluation.parsing import parse_tool_output
from tool_calling.evaluation.validation import validate_tool_output
from tool_calling.paths import PROJECT_ROOT


def score_tool_output(raw_output, expected, tool_schemas):
    """分别记录输出格式、接口约束和参考目标匹配情况。"""
    # 参考目标本身必须合法，否则应修正数据，不能继续评分。
    reference_errors = validate_tool_output(expected, tool_schemas)
    if reference_errors:
        raise ValueError(
            "参考目标不符合接口约束：" + "；".join(reference_errors)
        )

    expected_call = expected["action"] == "call"

    result = {
        "raw_output": raw_output,
        "prediction": None,
        "json_object_valid": False,
        "parse_error": None,
        "interface_valid": False,
        "validation_errors": [],
        "action_matches": False,
        # 非调用样本不适用完整调用比较，用 None 表示。
        "call_matches_reference": False if expected_call else None,
        "message_for_review": None,
    }

    # 保留原始输出；格式失败时记录原因，仍然返回评分结果。
    try:
        prediction = parse_tool_output(raw_output)
    except ValueError as error:
        result["parse_error"] = str(error)
        return result

    result["prediction"] = prediction
    result["json_object_valid"] = True

    validation_errors = validate_tool_output(prediction, tool_schemas)
    result["validation_errors"] = validation_errors
    result["interface_valid"] = not validation_errors

    # 动作正确与接口合法分别判断，便于定位错误。
    result["action_matches"] = (
        prediction.get("action") == expected["action"]
    )

    if expected_call:
        # 接口合法后，工具名称和全部参数也必须匹配参考目标。
        result["call_matches_reference"] = (
            result["interface_valid"] and prediction == expected
        )
    else:
        # 澄清和不支持的说明允许不同措辞，语义正确性需人工核对。
        result["message_for_review"] = prediction.get("message")

    return result


def main():
    tools = json.loads(
        (PROJECT_ROOT / "configs/tools.json").read_text(encoding="utf-8")
    )
    tool_schemas = {
        tool["function"]["name"]: tool["function"]["parameters"]
        for tool in tools
    }
    examples = json.loads(
        (PROJECT_ROOT / "data/manual/train_examples.json").read_text(
            encoding="utf-8"
        )
    )

    parser = argparse.ArgumentParser(
        description="对指定开发样例的原始输出进行评分。"
    )
    parser.add_argument("raw_output", help="需要评分的原始输出字符串。")
    parser.add_argument(
        "--example-id",
        default="train_002",
        choices=[item["id"] for item in examples],
    )
    args = parser.parse_args()

    example = next(
        item for item in examples if item["id"] == args.example_id
    )
    result = score_tool_output(
        args.raw_output, example["expected"], tool_schemas
    )

    print("用户请求：", example["query"])
    print(
        "参考目标：",
        json.dumps(example["expected"], ensure_ascii=False),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
import json


def reject_nonstandard_constant(value):
    """拒绝标准 JSON 不支持的 NaN、Infinity 和 -Infinity。"""
    raise ValueError(f"不允许非标准 JSON 数值：{value}")


def build_unique_object(pairs):
    """检查每个 JSON 对象的字段是否重复，避免静默覆盖。"""
    result = {}

    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON 对象出现重复字段：{key}")
        result[key] = value

    return result


def parse_tool_output(raw_output):
    """解析原始输出，要求单个 JSON 对象、唯一字段和标准数值。"""
    if not isinstance(raw_output, str):
        raise ValueError("原始输出必须是字符串。")

    # 直接解析完整原文，保留代码块、额外文字等格式问题。
    output = json.loads(
        raw_output,
        parse_constant=reject_nonstandard_constant,
        object_pairs_hook=build_unique_object,
    )

    if not isinstance(output, dict):
        raise ValueError("输出必须是单个 JSON 对象。")

    return output
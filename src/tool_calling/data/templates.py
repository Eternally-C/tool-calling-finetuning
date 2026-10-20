def fill_placeholders(value, variables):
    """递归替换 JSON 模板中的占位符。"""
    if isinstance(value, str):
        # 整个值都是占位符时，保留替换值的类型。
        # 后续生成整数参数时，也能避免把整数变成字符串。
        for name, replacement in variables.items():
            if value == "{" + name + "}":
                return replacement

        return value.format(**variables)

    if isinstance(value, dict):
        return {
            key: fill_placeholders(item, variables)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            fill_placeholders(item, variables)
            for item in value
        ]

    return value
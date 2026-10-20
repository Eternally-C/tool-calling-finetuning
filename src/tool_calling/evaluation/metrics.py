def get_predicted_action(record):
    """只识别成功解析后的有效动作名称。"""
    prediction = record["prediction"]
    if isinstance(prediction, dict):
        action = prediction.get("action")
        if action in ("call", "clarify", "unsupported"):
            return action

    return "invalid"


def print_rate(label, correct, total):
    """保留分子和分母，避免百分比掩盖样本范围。"""
    if total == 0:
        print(f"{label}：不适用，分母为 0")
    else:
        print(f"{label}：{correct}/{total}（{correct / total:.2%}）")
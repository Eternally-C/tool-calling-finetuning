# Train 与 Validation 共用生成设置，保持实验比较条件一致。
GENERATION_CONFIG = {
    "enable_thinking": False,
    "temperature": 0.0,
    "max_tokens": 256,
}


def select_smoke_samples(samples):
    """按固定顺序选取三类动作各一条，检查批量运行流程。"""
    selected = []

    for action in ("call", "clarify", "unsupported"):
        sample = next(
            (
                item
                for item in samples
                if item["expected"]["action"] == action
            ),
            None,
        )
        if sample is None:
            raise ValueError(f"候选数据中缺少 {action} 样本。")

        selected.append(sample)

    return selected
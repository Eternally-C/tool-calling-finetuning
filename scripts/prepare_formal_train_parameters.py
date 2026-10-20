import json
import random
from collections import Counter
from datetime import date, timedelta

from tool_calling.data.validation import validate_date_range
from tool_calling.paths import PROJECT_ROOT


OUTPUT_PATH = (
    PROJECT_ROOT / "data/generated/formal/train_parameter_sets.json"
)
SEED = 42


def build_parameter_sets():
    categories = ("electronics", "clothing", "home")
    metrics = ("revenue", "order_count", "units_sold")

    # 九种分类与指标组合重复三轮，再补三组，使两个枚举各出现十次。
    pairs = [
        (category, metric)
        for category in categories
        for metric in metrics
    ] * 3
    pairs.extend(zip(categories, metrics))

    # 固定局部随机种子，使参数顺序可复现，并打散枚举与日期的对应。
    random.Random(SEED).shuffle(pairs)

    parameter_sets = []
    for index, (category, metric) in enumerate(pairs):
        start = date(2024, 1, 1) + timedelta(days=12 * index)
        # 普通窗口包含六天；最后一组覆盖起止日期相同的合法情况。
        end = start + timedelta(days=0 if index == len(pairs) - 1 else 5)
        parameter_set = {
            "parameter_set_id": f"formal_train_params_{index + 1:03d}",
            "split": "train",
            # 合成编号只用于参数提取练习，不代表真实业务记录。
            "product_id": f"P{6101 + index}",
            "order_id": f"O{7101 + index}",
            "refund_id": f"R{8101 + index}",
            "category": category,
            "metric": metric,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "threshold": 3 + index,
        }
        validate_date_range(parameter_set)
        parameter_sets.append(parameter_set)

    return parameter_sets


def main():
    # 已生成的参数需要保留，避免重复运行覆盖后续人工调整。
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"参数文件已存在，未覆盖：{OUTPUT_PATH}")

    parameter_sets = build_parameter_sets()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(parameter_sets, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("参数组数量：", len(parameter_sets))
    print("商品分类分布：", dict(Counter(
        item["category"] for item in parameter_sets
    )))
    print("销售指标分布：", dict(Counter(
        item["metric"] for item in parameter_sets
    )))
    print("正整数阈值范围：", min(
        item["threshold"] for item in parameter_sets
    ), "至", max(item["threshold"] for item in parameter_sets))
    print("参数文件：", OUTPUT_PATH)


if __name__ == "__main__":
    main()
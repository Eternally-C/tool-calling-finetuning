import json
import random
from collections import Counter
from datetime import date, timedelta

from tool_calling.data.validation import validate_date_range
from tool_calling.paths import PROJECT_ROOT


OUTPUT_PATH = (
    PROJECT_ROOT / "data/generated/formal/test_parameter_sets.json"
)
SEED = 42


def build_parameter_sets():
    categories = ("electronics", "clothing", "home")
    metrics = ("revenue", "order_count", "units_sold")

    # 覆盖九种组合，每种分类和指标各出现十次。
    pairs = [
        (category, metric)
        for category in categories
        for metric in metrics
    ] * 3
    pairs.extend(zip(categories, metrics))

    # 使用局部随机种子，保证参数顺序可复现。
    random.Random(SEED).shuffle(pairs)

    parameter_sets = []
    for index, (category, metric) in enumerate(pairs):
        # 独立年份用于隔离正常日期，最后一组覆盖合法同日查询。
        start = date(2022, 1, 1) + timedelta(days=12 * index)
        end = start + timedelta(
            days=0 if index == len(pairs) - 1 else 5
        )

        parameter_set = {
            "parameter_set_id": f"formal_test_params_{index + 1:03d}",
            "split": "test",
            # 合成编号用于参数提取，不代表真实业务记录。
            "product_id": f"P{12101 + index}",
            "order_id": f"O{13101 + index}",
            "refund_id": f"R{14101 + index}",
            "category": category,
            "metric": metric,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "threshold": 81 + index,
        }
        validate_date_range(parameter_set)
        parameter_sets.append(parameter_set)

    return parameter_sets


def main():
    plan = json.loads(
        (PROJECT_ROOT / "configs/final_evaluation_plan.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        plan["status"] != "locked_before_test"
        or plan["test_policy"]["allow_test_tuning"] is not False
    ):
        raise ValueError("请先锁定最终评估方案。")

    if OUTPUT_PATH.exists():
        raise FileExistsError(f"参数文件已存在，未覆盖：{OUTPUT_PATH}")

    parameter_sets = build_parameter_sets()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("x", encoding="utf-8") as output_file:
        json.dump(
            parameter_sets, output_file, ensure_ascii=False, indent=2
        )
        output_file.write("\n")

    print("参数组数量：", len(parameter_sets))
    print("商品分类分布：", dict(Counter(
        item["category"] for item in parameter_sets
    )))
    print("销售指标分布：", dict(Counter(
        item["metric"] for item in parameter_sets
    )))
    print(
        "正整数阈值范围：",
        min(item["threshold"] for item in parameter_sets),
        "至",
        max(item["threshold"] for item in parameter_sets),
    )
    print(
        "有效日期范围：",
        parameter_sets[0]["start_date"],
        "至",
        parameter_sets[-1]["end_date"],
    )
    print("参数文件：", OUTPUT_PATH)


if __name__ == "__main__":
    main()
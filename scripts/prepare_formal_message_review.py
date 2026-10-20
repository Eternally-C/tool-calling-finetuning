import hashlib
import json
from collections import Counter

from tool_calling.paths import PROJECT_ROOT


SOURCE_RESULTS = {
    "baseline": (
        "results/formal/validation/"
        "validation_batch_20261017T010418_932868Z.jsonl",
        "8f349c585178b53676c3f94fe0a9522a7f38eb44f18440abd2635751f0e3d298",
    ),
    "step450": (
        "results/formal/validation/"
        "validation_lora_batch_20261017T024653_099789Z.jsonl",
        "755b20818e009ef59f3172293a37d896d026802bc31d82c78ba47fd0d2d3c343",
    ),
}
REVIEW_PATH = PROJECT_ROOT / (
    "results/formal/validation/message_review_baseline_step450.json"
)


def build_review():
    records_by_model = {}
    sources = {}
    for label, (relative_path, expected_hash) in SOURCE_RESULTS.items():
        source_bytes = (PROJECT_ROOT / relative_path).read_bytes()
        source_hash = hashlib.sha256(source_bytes).hexdigest()
        if source_hash != expected_hash:
            raise ValueError(f"{label} 原始结果的指纹不一致。")
        records = [
            json.loads(line)
            for line in source_bytes.decode("utf-8").splitlines()
        ]
        if len(records) != 200:
            raise ValueError(f"{label} 必须包含完整的 200 条结果。")
        if len({record["id"] for record in records}) != len(records):
            raise ValueError(f"{label} 存在重复样本编号。")

        run_info = records[0]["run_info"]
        if any(record["run_info"] != run_info for record in records):
            raise ValueError(f"{label} 混合了不同运行配置。")
        if (
            run_info.get("stage") != "formal"
            or run_info["sample_selection"] != "all"
            or any(record["split"] != "validation" for record in records)
        ):
            raise ValueError(f"{label} 必须是正式完整 Validation 结果。")
        records_by_model[label] = records
        sources[label] = {
            "source_result": relative_path,
            "source_result_sha256": source_hash,
            "run_info": run_info,
        }

    items = []
    for baseline, candidate in zip(
        records_by_model["baseline"],
        records_by_model["step450"],
        strict=True,
    ):
        for field in ("id", "query", "expected"):
            if baseline[field] != candidate[field]:
                raise ValueError(f"两个模型的样本来源不一致：{field}")

        expected_action = baseline["expected"]["action"]
        if expected_action not in ("clarify", "unsupported"):
            continue

        outputs = {}
        for label, record in (
            ("baseline", baseline),
            ("step450", candidate),
        ):
            prediction = record["prediction"]
            if not isinstance(prediction, dict):
                prediction = {}
            eligible = (
                record["interface_valid"] is True
                and prediction.get("action") == expected_action
            )
            # 自动检查已失败的样本仍保留，不能用文字评分抵消动作错误。
            outputs[label] = {
                "predicted_action": prediction.get("action"),
                "model_message": prediction.get("message"),
                "status": "pending" if eligible else "not_applicable",
                "reason": "" if eligible else "动作或接口未通过自动检查。",
                "suggested_message": None,
            }

        items.append({
            "id": baseline["id"],
            "query": baseline["query"],
            "expected_action": expected_action,
            "reference_message": baseline["expected"]["message"],
            "outputs": outputs,
        })

    if len(items) != 80:
        raise ValueError("应当导出 80 条参考非调用样本。")

    return {
        "review_version": 1,
        "stage": "formal",
        "review_type": "paired_non_call_messages",
        "sources": sources,
        "criteria": [
            "clarify 应指出实际需要补充、修正或澄清的信息。",
            "unsupported 应准确说明当前不支持的具体操作。",
            "说明不得编造业务事实、工具能力，或声称已完成操作。",
            "按语义判断，不要求与 reference_message 逐字一致。",
        ],
        "allowed_statuses": [
            "pending", "pass", "needs_improvement", "not_applicable"
        ],
        "items": items,
    }


def main():
    review = build_review()
    # 独占创建，防止覆盖已经填写的人工审查意见。
    with REVIEW_PATH.open("x", encoding="utf-8") as output_file:
        json.dump(review, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")

    print("已导出参考非调用样本：", len(review["items"]))
    for label in SOURCE_RESULTS:
        counts = Counter(
            item["outputs"][label]["status"]
            for item in review["items"]
        )
        print(
            label,
            "等待人工语义审查：", counts["pending"],
            "动作或接口错误：", counts["not_applicable"],
        )
    print("审查文件：", REVIEW_PATH)


if __name__ == "__main__":
    main()
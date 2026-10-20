import hashlib
import json
from collections import Counter
from pathlib import Path

from tool_calling.paths import PROJECT_ROOT


SOURCE_RESULT = PROJECT_ROOT / (
    "results/validation/"
    "validation_lora_batch_20261015T104456_776379Z.jsonl"
)
REVIEW_PATH = PROJECT_ROOT / "results/validation/message_review_step56.json"


def build_review():
    source_bytes = SOURCE_RESULT.read_bytes()
    records = [
        json.loads(line)
        for line in source_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("原始结果为空。")

    run_info = records[0]["run_info"]
    if any(record["run_info"] != run_info for record in records):
        raise ValueError("结果混合了不同运行配置。")
    if (
        run_info["mode"] != "validation_lora"
        or run_info["sample_selection"] != "all"
        or Path(run_info["checkpoint_path"]).name
        != "0000056_adapters.safetensors"
    ):
        raise ValueError("输入必须是 step 56 的完整 Validation 结果。")

    record_ids = [record["id"] for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("原始结果存在重复样本编号。")
    if any(record["split"] != "validation" for record in records):
        raise ValueError("原始结果包含其他数据分区。")

    items = []
    for record in records:
        expected_action = record["expected"]["action"]
        if expected_action not in ("clarify", "unsupported"):
            continue

        prediction = record["prediction"]
        if not isinstance(prediction, dict):
            prediction = {}
        eligible = (
            prediction.get("action") == expected_action
            and record["interface_valid"] is True
        )
        items.append({
            "id": record["id"],
            "query": record["query"],
            "expected_action": expected_action,
            "reference_message": record["expected"]["message"],
            "predicted_action": prediction.get("action"),
            "model_message": prediction.get("message"),
            # 动作或接口已失败时，单独保留，不计入说明文字审查分母。
            "status": "pending" if eligible else "not_applicable",
            "reason": "" if eligible else "动作或接口未通过自动检查。",
            "suggested_message": None,
        })

    return {
        "review_version": 1,
        # 保存原始结果指纹，人工意见与模型实际输出分别保留。
        "source_result": SOURCE_RESULT.relative_to(PROJECT_ROOT).as_posix(),
        "source_result_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "run_info": run_info,
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
    # 防止重复运行覆盖已经填写的人工审查意见。
    if REVIEW_PATH.exists():
        raise FileExistsError(f"审查文件已存在，未覆盖：{REVIEW_PATH}")

    review = build_review()
    REVIEW_PATH.write_text(
        json.dumps(review, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    counts = Counter(item["status"] for item in review["items"])
    print("已导出参考非调用样本：", len(review["items"]))
    print("等待人工语义审查：", counts["pending"])
    print("动作或接口错误，语义审查不适用：", counts["not_applicable"])
    print("审查文件：", REVIEW_PATH)


if __name__ == "__main__":
    main()
import argparse
import hashlib
import json
from pathlib import Path

from tool_calling.evaluation.metrics import get_predicted_action, print_rate


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(
        description="汇总完整 Train 候选批量运行的结果。"
    )
    parser.add_argument(
        "result_file",
        type=Path,
        help="需要汇总的 train_batch JSONL 文件。",
    )
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in args.result_file.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    if not records:
        raise ValueError("结果文件为空。")

    record_ids = [record["id"] for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("结果文件中出现重复样本编号。")

    run_info = records[0]["run_info"]
    if any(record["run_info"] != run_info for record in records):
        raise ValueError("结果文件混合了不同运行配置。")

    if (
        run_info["mode"] != "train_development"
        or run_info["sample_selection"] != "all"
    ):
        raise ValueError("本脚本仅汇总完整 Train 候选批量结果。")

    # 核对数据版本与完整性，避免把中断后的部分结果当成完整运行。
    source_text = (
        PROJECT_ROOT / "data/generated/train_preview.jsonl"
    ).read_text(encoding="utf-8")
    source_hash = hashlib.sha256(
        source_text.encode("utf-8")
    ).hexdigest()
    if run_info["data_sha256"] != source_hash:
        raise ValueError(
            "结果与当前 Train 候选数据版本不同，请先核对源文件。"
        )

    samples = [
        json.loads(line)
        for line in source_text.splitlines()
    ]
    if record_ids != [sample["id"] for sample in samples]:
        raise ValueError("结果样本不完整，或顺序与源数据不一致。")

    for record, sample in zip(records, samples):
        if record["split"] != "train":
            raise ValueError("结果文件中出现了其他分区的样本。")

        if any(record[key] != value for key, value in sample.items()):
            raise ValueError(f"来源字段不一致：{record['id']}")

        for key in (
            "json_object_valid",
            "interface_valid",
            "action_matches",
        ):
            if type(record[key]) is not bool:
                raise ValueError(f"评分字段 {key} 必须是布尔值。")

        # call 使用布尔值；其他参考动作使用 None 表示不适用。
        if record["expected"]["action"] == "call":
            if type(record["call_matches_reference"]) is not bool:
                raise ValueError("call 样本缺少完整调用比较结果。")
        elif record["call_matches_reference"] is not None:
            raise ValueError("非 call 样本的完整调用比较必须是不适用。")

    total = len(records)
    expected_calls = [
        record
        for record in records
        if record["expected"]["action"] == "call"
    ]

    # 无法解析的输出仍计入全部样本，但不能识别为预测 call。
    predicted_calls = [
        record
        for record in records
        if get_predicted_action(record) == "call"
    ]
    correct_calls = sum(
        record["call_matches_reference"] is True
        for record in expected_calls
    )

    print("结果文件：", args.result_file)
    print("已核对完整 Train 候选记录：", total)
    print_rate(
        "JSON 对象格式正确率",
        sum(record["json_object_valid"] is True for record in records),
        total,
    )
    print_rate(
        "接口约束通过率",
        sum(record["interface_valid"] is True for record in records),
        total,
    )
    print_rate(
        "全部样本的动作匹配率",
        sum(record["action_matches"] is True for record in records),
        total,
    )
    print_rate(
        "参考 call 样本的完整调用匹配率",
        correct_calls,
        len(expected_calls),
    )
    print_rate(
        "已解析为 call 的输出中，完整调用正确率",
        correct_calls,
        len(predicted_calls),
    )

    # 分别统计三类参考动作，定位整体指标掩盖的薄弱类别。
    for action in ("call", "clarify", "unsupported"):
        group = [
            record
            for record in records
            if record["expected"]["action"] == action
        ]
        print_rate(
            f"参考 {action} 样本的动作匹配率",
            sum(record["action_matches"] is True for record in group),
            len(group),
        )

    failures = [
        record
        for record in records
        if (
            not record["json_object_valid"]
            or not record["interface_valid"]
            or not record["action_matches"]
            or record["call_matches_reference"] is False
        )
    ]
    print("自动检查发现的问题样本：", len(failures))
    for record in failures:
        print(
            record["id"],
            "| 参考动作：", record["expected"]["action"],
            "| 预测动作：", get_predicted_action(record),
            "| 类型：", record["case_type"],
        )

    print("说明内容的语义质量仍需人工核对。")


if __name__ == "__main__":
    main()
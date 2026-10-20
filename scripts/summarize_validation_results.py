import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from tool_calling.evaluation.scoring import score_tool_output
from tool_calling.evaluation.metrics import get_predicted_action, print_rate
from tool_calling.modeling.generation import GENERATION_CONFIG


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(
        description="核对并汇总完整 Validation 基线或 LoRA 结果。"
    )
    parser.add_argument("result_file", type=Path)
    args = parser.parse_args()
    result_path = args.result_file.resolve()

    records = [
        json.loads(line)
        for line in result_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("结果文件为空。")

    record_ids = [record["id"] for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("结果中出现重复样本编号。")

    run_info = records[0]["run_info"]
    if any(record["run_info"] != run_info for record in records):
        raise ValueError("结果混合了不同运行配置。")
    if run_info["sample_selection"] != "all":
        raise ValueError("本脚本只汇总完整 Validation 结果。")

    expected_model = str(PROJECT_ROOT / "models/qwen3_8b_4bit")
    if (
        run_info["model_path"] != expected_model
        or run_info["generation_config"] != GENERATION_CONFIG
    ):
        raise ValueError("基座模型路径或生成设置与当前项目不一致。")

    mode = run_info["mode"]
    if mode == "validation_baseline":
        if (
            run_info.get("adapter_path") is not None
            or run_info.get("checkpoint_path") is not None
        ):
            raise ValueError("基线结果不能记录已加载的 adapter 或 checkpoint。")

    elif mode == "validation_lora":
        checkpoint_value = run_info.get("checkpoint_path")
        if not isinstance(checkpoint_value, str) or not checkpoint_value:
            raise ValueError("LoRA 结果必须记录 checkpoint 文件路径。")

        checkpoint = Path(checkpoint_value).resolve()
        if not checkpoint.is_file() or checkpoint.suffix != ".safetensors":
            raise ValueError(f"checkpoint 文件无效：{checkpoint}")

        # 核对记录对应的实际权重、训练配置和 Train 数据版本。
        config_bytes = (
            checkpoint.parent / "adapter_config.json"
        ).read_bytes()
        train_bytes = (
            PROJECT_ROOT / "data/processed/pilot_sft/train.jsonl"
        ).read_bytes()
        expected_info = {
            "adapter_path": str(checkpoint.parent),
            "checkpoint_sha256": hashlib.sha256(
                checkpoint.read_bytes()
            ).hexdigest(),
            "adapter_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "train_data_sha256": hashlib.sha256(train_bytes).hexdigest(),
        }
        for field, value in expected_info.items():
            if run_info.get(field) != value:
                raise ValueError(f"LoRA 结果的 {field} 核对失败。")

        adapter_config = json.loads(config_bytes.decode("utf-8"))
        if (
            adapter_config["fine_tune_type"] != "lora"
            or adapter_config["model"] != run_info["model_path"]
            or adapter_config["train_data_sha256"] != run_info["train_data_sha256"]
        ):
            raise ValueError("adapter 配置与结果记录不一致。")

    else:
        raise ValueError(f"不支持的 Validation 运行类型：{mode}")

    # 核对源数据版本和完整顺序，防止把部分结果当成完整实验。
    source_text = (
        PROJECT_ROOT / "data/generated/validation_preview.jsonl"
    ).read_text(encoding="utf-8")
    source_hash = hashlib.sha256(
        source_text.encode("utf-8")
    ).hexdigest()
    if run_info["data_sha256"] != source_hash:
        raise ValueError("结果与当前 Validation 候选数据版本不同。")

    samples = [
        json.loads(line)
        for line in source_text.splitlines()
        if line.strip()
    ]
    if record_ids != [sample["id"] for sample in samples]:
        raise ValueError("结果样本不完整，或顺序与源数据不一致。")

    tools = json.loads(
        (PROJECT_ROOT / "configs/tools.json").read_text(encoding="utf-8")
    )
    tool_schemas = {
        tool["function"]["name"]: tool["function"]["parameters"]
        for tool in tools
    }
    rules = (PROJECT_ROOT / "configs/system_prompt.txt").read_text(
        encoding="utf-8"
    )
    system_content = (
        rules.strip()
        + "\n\n可用工具定义：\n"
        + json.dumps(tools, ensure_ascii=False, indent=2)
    )
    system_hash = hashlib.sha256(
        system_content.encode("utf-8")
    ).hexdigest()
    if run_info["system_content_sha256"] != system_hash:
        raise ValueError("结果与当前提示词或工具定义版本不同。")

    # 使用保存的原始输出重新评分；这里不会加载或运行模型。
    for record, sample in zip(records, samples):
        if record["split"] != "validation":
            raise ValueError("结果中出现了其他分区。")
        if any(record[key] != value for key, value in sample.items()):
            raise ValueError(f"来源字段不一致：{record['id']}")

        rescored = score_tool_output(
            record["raw_output"], sample["expected"], tool_schemas
        )
        if any(record[key] != value for key, value in rescored.items()):
            raise ValueError(f"保存的评分与重新计算不一致：{record['id']}")

    total = len(records)
    expected_calls = [
        record for record in records
        if record["expected"]["action"] == "call"
    ]
    predicted_calls = [
        record for record in records
        if get_predicted_action(record) == "call"
    ]
    correct_calls = sum(
        record["call_matches_reference"] is True
        for record in expected_calls
    )

    print("结果文件：", result_path)
    print("运行类型：", mode)
    print("checkpoint：", run_info.get("checkpoint_path") or "基线模型")
    print("已核对完整 Validation 记录：", total)

    for label, field in (
        ("JSON 对象格式正确率", "json_object_valid"),
        ("接口约束通过率", "interface_valid"),
        ("全部样本的动作匹配率", "action_matches"),
    ):
        print_rate(
            label,
            sum(record[field] is True for record in records),
            total,
        )

    # 两项调用指标的分母不同，分别反映应调用与实际调用的表现。
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

    for action in ("call", "clarify", "unsupported"):
        group = [
            record for record in records
            if record["expected"]["action"] == action
        ]
        print_rate(
            f"参考 {action} 样本的动作匹配率",
            sum(record["action_matches"] is True for record in group),
            len(group),
        )

    failures = [
        record for record in records
        if (
            not record["json_object_valid"]
            or not record["interface_valid"]
            or not record["action_matches"]
            or record["call_matches_reference"] is False
        )
    ]
    # 非调用说明的语义仍需人工核对，此处仅统计自动检查。
    print_rate("全部自动检查通过率", total - len(failures), total)
    print("自动检查发现的问题样本：", len(failures))
    print(
        "错误模板分布：",
        dict(Counter(record["template_id"] for record in failures)),
    )
    for record in failures:
        print(
            record["id"],
            "| 参考动作：", record["expected"]["action"],
            "| 预测动作：", get_predicted_action(record),
        )

    print("说明内容的语义质量仍需人工核对。")


if __name__ == "__main__":
    main()
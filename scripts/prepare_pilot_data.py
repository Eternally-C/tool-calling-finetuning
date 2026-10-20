import hashlib
import json
from collections import Counter
from pathlib import Path

from tool_calling.data.isolation import check_parameter_isolation
from tool_calling.evaluation.validation import validate_tool_output


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_FIELDS = (
    "id",
    "template_id",
    "family_id",
    "parameter_set_id",
    "split",
    "case_type",
)


def main():
    check_parameter_isolation()

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

    samples_by_split = {}
    converted = {}

    for split in ("train", "validation"):
        source_path = (
            PROJECT_ROOT / f"data/generated/{split}_preview.jsonl"
        )
        source_bytes = source_path.read_bytes()
        source_hash = hashlib.sha256(source_bytes).hexdigest()
        samples = [
            json.loads(line)
            for line in source_bytes.decode("utf-8").splitlines()
            if line.strip()
        ]

        if not samples:
            raise ValueError(f"{split} 数据为空。")
        if len({sample["id"] for sample in samples}) != len(samples):
            raise ValueError(f"{split} 样本编号重复。")

        samples_by_split[split] = samples
        converted[split] = []

        for sample in samples:
            if sample["split"] != split:
                raise ValueError(f"样本分区不一致：{sample['id']}")
            if (
                not isinstance(sample["query"], str)
                or not sample["query"].strip()
            ):
                raise ValueError(f"用户请求为空：{sample['id']}")

            errors = validate_tool_output(sample["expected"], tool_schemas)
            if errors:
                raise ValueError(
                    f"参考答案不合法：{sample['id']}，{errors}"
                )

            # 元数据用于追溯来源，不放入模型的聊天消息。
            metadata = {
                field: sample[field] for field in METADATA_FIELDS
            }
            metadata["source_sha256"] = source_hash
            metadata["system_content_sha256"] = system_hash

            converted[split].append({
                "messages": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": sample["query"]},
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            sample["expected"], ensure_ascii=False
                        ),
                    },
                ],
                "metadata": metadata,
            })

    # 写入前检查两个分区的编号、请求和已标记的表达家族。
    for field in ("id", "query", "family_id"):
        train_values = {
            sample[field] for sample in samples_by_split["train"]
        }
        validation_values = {
            sample[field] for sample in samples_by_split["validation"]
        }
        if train_values & validation_values:
            raise ValueError(f"{field} 在 Train / Validation 中重叠。")

    output_dir = PROJECT_ROOT / "data/processed/pilot_sft"
    output_paths = {
        "train": output_dir / "train.jsonl",
        "validation": output_dir / "valid.jsonl",
    }

    # 试训目录仅准备 Train / Validation，避免混入最终 Test。
    if (output_dir / "test.jsonl").exists():
        raise ValueError("试训数据目录中不能包含 test.jsonl。")
    for path in output_paths.values():
        if path.exists():
            raise FileExistsError(f"文件已存在，停止以避免覆盖：{path}")

    # 所有数据检查通过后再生成文件；保留原始样本顺序。
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, path in output_paths.items():
        with path.open("x", encoding="utf-8") as file:
            for record in converted[split]:
                file.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

        action_counts = Counter(
            sample["expected"]["action"]
            for sample in samples_by_split[split]
        )
        print(f"{split}：{len(converted[split])} 条")
        print("动作分布：", dict(action_counts))
        print("输出文件：", path)

    print("试训聊天数据生成完成。")


if __name__ == "__main__":
    main()
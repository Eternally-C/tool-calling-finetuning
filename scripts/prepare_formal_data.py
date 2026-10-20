import hashlib
import json
from collections import Counter

from tool_calling.data.formal_isolation import check_formal_data_isolation
from tool_calling.paths import PROJECT_ROOT


OUTPUT_DIR = PROJECT_ROOT / "data/processed/formal_sft"
METADATA_FIELDS = (
    "id",
    "template_id",
    "family_id",
    "parameter_set_id",
    "split",
    "case_type",
)


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_formal_chat_data():
    samples_by_split = check_formal_data_isolation()
    tool_path = PROJECT_ROOT / "configs/tools.json"
    tools = json.loads(tool_path.read_bytes())
    rules = (PROJECT_ROOT / "configs/system_prompt.txt").read_text(
        encoding="utf-8"
    )
    # 保持与已有基线一致的系统规则和工具定义。
    system_content = (
        rules.strip()
        + "\n\n可用工具定义：\n"
        + json.dumps(tools, ensure_ascii=False, indent=2)
    )
    system_hash = hashlib.sha256(
        system_content.encode("utf-8")
    ).hexdigest()
    plan_hash = file_sha256(
        PROJECT_ROOT / "configs/formal_data_plan.json"
    )
    converted = {}

    for split, samples in samples_by_split.items():
        source_path = (
            PROJECT_ROOT / f"data/generated/formal/{split}_selected.jsonl"
        )
        source_bytes = source_path.read_bytes()
        # 指纹与检查过的样本必须对应同一份文件内容。
        current_samples = [
            json.loads(line)
            for line in source_bytes.decode("utf-8").splitlines()
        ]
        if current_samples != samples:
            raise ValueError(f"{split} 选样文件在检查期间发生变化。")
        source_hash = hashlib.sha256(source_bytes).hexdigest()
        expected_source = {
            "stage": "formal",
            "templates_sha256": file_sha256(
                PROJECT_ROOT / f"data/manual/formal/{split}_templates.json"
            ),
            "parameter_sets_sha256": file_sha256(
                PROJECT_ROOT
                / f"data/generated/formal/{split}_parameter_sets.json"
            ),
            "tools_sha256": file_sha256(tool_path),
        }
        candidate_hash = file_sha256(
            PROJECT_ROOT / f"data/generated/formal/{split}_candidates.jsonl"
        )
        converted[split] = []

        for sample in samples:
            selection = sample["selection_info"]
            if (
                sample["source_info"] != expected_source
                or selection["candidates_sha256"] != candidate_hash
                or selection["plan_sha256"] != plan_hash
            ):
                raise ValueError(f"来源版本不一致：{sample['id']}")

            # 元数据只用于追溯，不进入模型的对话输入。
            metadata = {
                field: sample[field] for field in METADATA_FIELDS
            }
            metadata.update({
                "stage": "formal",
                "source_sha256": source_hash,
                "system_content_sha256": system_hash,
                "source_info": sample["source_info"],
                "selection_info": selection,
            })
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

    return converted


def main():
    output_paths = {
        "train": OUTPUT_DIR / "train.jsonl",
        "validation": OUTPUT_DIR / "valid.jsonl",
    }
    # 此阶段只准备训练与验证数据，防止最终 Test 混入训练目录。
    if (OUTPUT_DIR / "test.jsonl").exists():
        raise ValueError("正式训练目录中不能包含 test.jsonl。")
    for path in output_paths.values():
        if path.exists():
            raise FileExistsError(f"文件已存在，未覆盖：{path}")

    converted = build_formal_chat_data()
    # 所有检查通过后再写入，保持选样文件中的顺序。
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split, path in output_paths.items():
        records = converted[split]
        with path.open("x", encoding="utf-8") as output_file:
            for record in records:
                output_file.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

        counts = Counter(
            json.loads(record["messages"][-1]["content"])["action"]
            for record in records
        )
        print(f"{split}：{len(records)} 条")
        print("动作分布：", dict(counts))
        print("输出文件：", path)

    print("正式聊天数据生成完成。")


if __name__ == "__main__":
    main()
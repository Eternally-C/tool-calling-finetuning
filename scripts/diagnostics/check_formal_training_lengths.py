import argparse
import json
from statistics import median

from mlx_lm.tokenizer_utils import load as load_tokenizer
from mlx_lm.tuner.datasets import ChatDataset

from tool_calling.evaluation.validation import validate_tool_output
from tool_calling.modeling.tokenization import set_non_thinking_default
from tool_calling.paths import PROJECT_ROOT


def print_lengths(label, values):
    print(
        f"{label}：最短 {min(values)}，"
        f"中位数 {median(values)}，最长 {max(values)}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="检查正式 SFT 数据的长度与监督边界。"
    )
    parser.add_argument("--max-seq-length", type=int, default=2048)
    args = parser.parse_args()
    if args.max_seq_length < 2:
        raise ValueError("长度上限必须至少为 2。")

    data_dir = PROJECT_ROOT / "data/processed/formal_sft"
    if (data_dir / "test.jsonl").exists():
        raise ValueError("正式训练目录中不能包含 test.jsonl。")
    tools = json.loads(
        (PROJECT_ROOT / "configs/tools.json").read_text(encoding="utf-8")
    )
    tool_schemas = {
        tool["function"]["name"]: tool["function"]["parameters"]
        for tool in tools
    }

    # 仅加载已有的本地分词器，不加载模型权重。
    model_path = PROJECT_ROOT / "models/qwen3_8b_4bit"
    model_config = json.loads(
        (model_path / "config.json").read_text(encoding="utf-8")
    )
    tokenizer = load_tokenizer(
        model_path,
        tokenizer_config_extra={
            "local_files_only": True,
            "trust_remote_code": False,
        },
        eos_token_ids=model_config.get("eos_token_id"),
    )
    tokenizer = set_non_thinking_default(tokenizer)

    oversized = []
    print("本次检查的完整序列长度上限：", args.max_seq_length)
    for split, filename in (
        ("train", "train.jsonl"),
        ("validation", "valid.jsonl"),
    ):
        records = [
            json.loads(line)
            for line in (data_dir / filename).read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        if not records:
            raise ValueError(f"{split} 数据为空。")
        ids = [record["metadata"]["id"] for record in records]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{split} 样本编号重复。")

        dataset = ChatDataset(records, tokenizer, mask_prompt=True)
        prompt_lengths, full_lengths, target_lengths = [], [], []
        for record in records:
            metadata = record["metadata"]
            sample_id = metadata["id"]
            if set(record) != {"messages", "metadata"}:
                raise ValueError(f"记录字段不符合约定：{sample_id}")
            if (
                metadata["stage"] != "formal"
                or metadata["split"] != split
            ):
                raise ValueError(f"阶段或分区不一致：{sample_id}")

            messages = record["messages"]
            if [message["role"] for message in messages] != [
                "system", "user", "assistant"
            ]:
                raise ValueError(f"消息角色不符合约定：{sample_id}")
            expected = json.loads(messages[-1]["content"])
            errors = validate_tool_output(expected, tool_schemas)
            if errors:
                raise ValueError(f"参考目标不合法：{sample_id}，{errors}")

            # 使用实际训练记录；ChatDataset 不把 metadata 加入对话。
            tokens, offset = dataset.process(record)
            # 推理提示词不含参考答案，与训练序列前缀逐 token 比较。
            prompt_tokens = tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if offset != len(prompt_tokens) or tokens[:offset] != prompt_tokens:
                raise ValueError(f"监督边界不一致：{sample_id}")
            target_length = len(tokens) - offset
            if target_length <= 0:
                raise ValueError(f"监督目标为空：{sample_id}")

            prompt_lengths.append(offset)
            full_lengths.append(len(tokens))
            target_lengths.append(target_length)
            if len(tokens) > args.max_seq_length:
                oversized.append((split, sample_id, len(tokens)))

        print(f"{split}：{len(records)} 条")
        print_lengths("提示词长度", prompt_lengths)
        print_lengths("完整序列长度", full_lengths)
        print_lengths("监督目标长度", target_lengths)
        print("监督边界一致：", len(records), "/", len(records))
        print(
            "超过长度上限的样本数：",
            sum(length > args.max_seq_length for length in full_lengths),
        )
        longest_index = full_lengths.index(max(full_lengths))
        print("最长样本编号：", ids[longest_index])

    if oversized:
        for split, sample_id, length in oversized[:5]:
            print("超长样本：", split, sample_id, length)
        raise ValueError(f"共有 {len(oversized)} 条超长样本，请先处理。")
    print("全部正式样本的长度与监督边界检查通过。")


if __name__ == "__main__":
    main()
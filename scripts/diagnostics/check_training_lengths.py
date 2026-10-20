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
        description="检查 Train / Validation 候选的完整长度和监督边界。"
    )
    parser.add_argument("--max-seq-length", type=int, default=2048)
    args = parser.parse_args()
    if args.max_seq_length < 2:
        raise ValueError("长度上限必须至少为 2。")

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

    for split in ("train", "validation"):
        source_path = (
            PROJECT_ROOT / f"data/generated/{split}_preview.jsonl"
        )
        samples = [
            json.loads(line)
            for line in source_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        if not samples:
            raise ValueError(f"{split} 候选数据为空。")
        if len({sample["id"] for sample in samples}) != len(samples):
            raise ValueError(f"{split} 候选编号重复。")

        prompt_lengths = []
        full_lengths = []
        target_lengths = []

        for sample in samples:
            if sample["split"] != split:
                raise ValueError(f"样本分区不一致：{sample['id']}")

            errors = validate_tool_output(sample["expected"], tool_schemas)
            if errors:
                raise ValueError(
                    f"参考目标不合法：{sample['id']}，{errors}"
                )

            # 参考答案只放在 assistant 消息中。
            prompt_messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": sample["query"]},
            ]
            messages = prompt_messages + [
                {
                    "role": "assistant",
                    "content": json.dumps(
                        sample["expected"], ensure_ascii=False
                    ),
                }
            ]
            item = {"messages": messages}
            dataset = ChatDataset([item], tokenizer, mask_prompt=True)
            tokens, offset = dataset.process(item)

            # 与实际基线推理的提示词逐 token 核对。
            prompt_tokens = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if (
                offset != len(prompt_tokens)
                or tokens[:offset] != prompt_tokens
            ):
                raise ValueError(f"监督边界不一致：{sample['id']}")

            target_length = len(tokens) - offset
            if target_length <= 0:
                raise ValueError(f"监督目标为空：{sample['id']}")

            prompt_lengths.append(offset)
            full_lengths.append(len(tokens))
            target_lengths.append(target_length)

            # 检查完整序列，避免截断掉部分或全部参考答案。
            if len(tokens) > args.max_seq_length:
                oversized.append((split, sample["id"], len(tokens)))

        print(f"{split}：{len(samples)} 条")
        print_lengths("提示词长度", prompt_lengths)
        print_lengths("完整序列长度", full_lengths)
        print_lengths("监督目标长度", target_lengths)
        print("监督边界一致：", len(samples), "/", len(samples))
        print(
            "超过长度上限的样本数：",
            sum(length > args.max_seq_length for length in full_lengths),
        )
        longest_index = full_lengths.index(max(full_lengths))
        print("最长样本编号：", samples[longest_index]["id"])

    if oversized:
        for split, sample_id, length in oversized[:5]:
            print("超长样本：", split, sample_id, length)
        raise ValueError(
            f"共有 {len(oversized)} 条样本超过上限，需要先调整长度设置。"
        )

    print("全部候选样本的长度与监督边界检查通过。")


if __name__ == "__main__":
    main()
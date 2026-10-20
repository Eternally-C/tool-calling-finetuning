import json

from mlx_lm.tokenizer_utils import load as load_tokenizer
from mlx_lm.tuner.datasets import ChatDataset

from tool_calling.evaluation.validation import validate_tool_output
from tool_calling.modeling.tokenization import set_non_thinking_default
from tool_calling.paths import PROJECT_ROOT


def main():
    source_path = PROJECT_ROOT / "data/generated/train_preview.jsonl"
    sample = next(
        json.loads(line)
        for line in source_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if sample["split"] != "train":
        raise ValueError("本次检查只允许使用 Train 样本。")

    tools = json.loads(
        (PROJECT_ROOT / "configs/tools.json").read_text(encoding="utf-8")
    )
    tool_schemas = {
        tool["function"]["name"]: tool["function"]["parameters"]
        for tool in tools
    }
    errors = validate_tool_output(sample["expected"], tool_schemas)
    if errors:
        raise ValueError(f"参考输出不合法：{errors}")

    rules = (PROJECT_ROOT / "configs/system_prompt.txt").read_text(
        encoding="utf-8"
    )
    system_content = (
        rules.strip()
        + "\n\n可用工具定义：\n"
        + json.dumps(tools, ensure_ascii=False, indent=2)
    )
    target_text = json.dumps(sample["expected"], ensure_ascii=False)

    # 推理输入只包含规则和请求；训练时另加助手参考答案。
    prompt_messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": sample["query"]},
    ]
    messages = prompt_messages + [
        {"role": "assistant", "content": target_text}
    ]

    model_path = PROJECT_ROOT / "models/qwen3_8b_4bit"
    model_config = json.loads(
        (model_path / "config.json").read_text(encoding="utf-8")
    )

    # 只加载本地 tokenizer，保持与 MLX-LM 使用的分词器一致。
    tokenizer = load_tokenizer(
        model_path,
        tokenizer_config_extra={
            "local_files_only": True,
            "trust_remote_code": False,
        },
        eos_token_ids=model_config.get("eos_token_id"),
    )

    prompt_tokens = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    full_tokens = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )

    # 答案起点之前的 token 必须与实际推理提示词完全一致。
    prefix_matches = full_tokens[:len(prompt_tokens)] == prompt_tokens
    print("样本编号：", sample["id"])
    print("用户请求：", sample["query"])
    print("助手参考答案：", target_text)
    print("基线推理提示词 token 数：", len(prompt_tokens))
    print("完整训练序列 token 数：", len(full_tokens))
    print("完整序列以基线提示词开头：", prefix_matches)

    if not prefix_matches:
        raise ValueError("训练序列与基线提示词前缀不一致，需要先核查模板。")

    print(
        "按基线提示词边界截取的学习目标：",
        repr(tokenizer.decode(
            full_tokens[len(prompt_tokens):],
            skip_special_tokens=False,
        )),
    )

    # 对照本机 MLX-LM 默认 ChatDataset 的提示词遮罩边界。
    item = {"messages": messages}
    dataset = ChatDataset([item], tokenizer, mask_prompt=True)
    default_tokens, default_offset = dataset.process(item)

    print("MLX-LM 默认完整序列是否相同：", default_tokens == full_tokens)
    print("MLX-LM 默认答案起点：", default_offset)
    print("与基线提示词长度一致：", default_offset == len(prompt_tokens))
    print(
        "MLX-LM 默认计入损失的目标：",
        repr(tokenizer.decode(
            default_tokens[default_offset:],
            skip_special_tokens=False,
        )),
    )

    # 设置进程内默认值，使 ChatDataset 的模板调用关闭 thinking。
    tokenizer = set_non_thinking_default(tokenizer)
    aligned_dataset = ChatDataset([item], tokenizer, mask_prompt=True)
    aligned_tokens, aligned_offset = aligned_dataset.process(item)

    print("对齐后完整序列一致：", aligned_tokens == full_tokens)
    print("对齐后答案起点：", aligned_offset)
    print(
        "对齐后起点与基线一致：",
        aligned_offset == len(prompt_tokens),
    )

    if (
        aligned_tokens != full_tokens
        or aligned_offset != len(prompt_tokens)
    ):
        raise ValueError("设置默认值后，训练格式仍未与基线对齐。")

    print(
        "对齐后计入损失的目标：",
        repr(tokenizer.decode(
            aligned_tokens[aligned_offset:],
            skip_special_tokens=False,
        )),
    )


if __name__ == "__main__":
    main()
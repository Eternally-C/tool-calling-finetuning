import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler

from tool_calling.data.isolation import check_parameter_isolation
from tool_calling.modeling.loading import load_model_with_checkpoint
from tool_calling.modeling.generation import (
    GENERATION_CONFIG,
    select_smoke_samples,
)
from tool_calling.evaluation.scoring import score_tool_output
from tool_calling.evaluation.validation import validate_tool_output


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(
        description="运行 Validation 候选样本，默认三类动作各一条。"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="运行全部 Validation 候选样本。",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="需要评估的 checkpoint 文件；不提供时运行基线。",
    )
    args = parser.parse_args()

    # 开始推理前，重新检查当前 Train / Validation 参数池。
    check_parameter_isolation()

    sample_path = PROJECT_ROOT / "data/generated/validation_preview.jsonl"
    sample_text = sample_path.read_text(encoding="utf-8")
    samples = [
        json.loads(line)
        for line in sample_text.splitlines()
        if line.strip()
    ]

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

    if not samples:
        raise ValueError("Validation 候选样本文件为空。")

    sample_ids = [sample["id"] for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("候选样本编号重复。")

    # 加载模型前检查分区和参考目标。
    for sample in samples:
        if sample["split"] != "validation":
            raise ValueError("本脚本只允许读取 Validation 候选样本。")

        errors = validate_tool_output(sample["expected"], tool_schemas)
        if errors:
            raise ValueError(
                f"参考目标不合法：{sample['id']}，{errors}"
            )

    # 复用已有选样函数和生成设置，保持 Train / Validation 推理一致。
    if not args.full:
        samples = select_smoke_samples(samples)

    system_content = (
        rules.strip()
        + "\n\n可用工具定义：\n"
        + json.dumps(tools, ensure_ascii=False, indent=2)
    )
    model_path = PROJECT_ROOT / "models/qwen3_8b_4bit"

    checkpoint = args.checkpoint
    if checkpoint is not None:
        checkpoint = checkpoint.expanduser()
        if not checkpoint.is_absolute():
            checkpoint = PROJECT_ROOT / checkpoint

    print("本次样本数量：", len(samples))
    print("正在加载本地模型及指定权重……", flush=True)

    # 只加载一次模型；各条请求仍使用独立对话。
    model, tokenizer, adapter_info = load_model_with_checkpoint(
        model_path, checkpoint
    )
    print(
        "本次 checkpoint：",
        adapter_info["checkpoint_path"] or "未加载，使用基线模型",
    )

    # 记录实际加载文件及哈希，区分不同训练轮次。
    run_info = {
        "mode": (
            "validation_lora"
            if checkpoint is not None
            else "validation_baseline"
        ),
        "sample_selection": "all" if args.full else "first_per_action",
        "model_path": str(model_path),
        **adapter_info,
        "generation_config": GENERATION_CONFIG,
        "data_sha256": hashlib.sha256(
            sample_text.encode("utf-8")
        ).hexdigest(),
        "system_content_sha256": hashlib.sha256(
            system_content.encode("utf-8")
        ).hexdigest(),
    }

    output_dir = PROJECT_ROOT / "results/validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )
    if args.checkpoint is None:
        prefix = "validation_batch" if args.full else "validation_smoke"
    else:
        prefix = (
            "validation_lora_batch" if args.full else "validation_lora_smoke"
        )
    output_path = output_dir / f"{prefix}_{timestamp}.jsonl"

    # 创建独立结果文件，逐条保存，避免覆盖此前的运行记录。
    with output_path.open("x", encoding="utf-8") as output_file:
        for index, sample in enumerate(samples, start=1):
            # 参考答案只用于评分，不进入模型输入。
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": sample["query"]},
            ]
            prompt_tokens = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                enable_thinking=GENERATION_CONFIG["enable_thinking"],
            )

            print(
                f"[{index}/{len(samples)}] {sample['id']}",
                flush=True,
            )
            response = generate(
                model,
                tokenizer,
                prompt=prompt_tokens,
                max_tokens=GENERATION_CONFIG["max_tokens"],
                sampler=make_sampler(
                    temp=GENERATION_CONFIG["temperature"]
                ),
                verbose=False,
            )

            # 模型答错或格式失败也保留记录，不能跳过失败样本。
            score = score_tool_output(
                response, sample["expected"], tool_schemas
            )
            record = {
                **sample,
                "run_info": run_info,
                "input_tokens": len(prompt_tokens),
                **score,
            }
            output_file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )
            output_file.flush()

            print(
                "接口通过：", score["interface_valid"],
                "动作匹配：", score["action_matches"],
                "完整调用匹配：", score["call_matches_reference"],
                flush=True,
            )

    print("本次运行完成。")
    print("结果文件：", output_path)


if __name__ == "__main__":
    main()
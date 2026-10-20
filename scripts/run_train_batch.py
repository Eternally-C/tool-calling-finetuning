import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from tool_calling.evaluation.scoring import score_tool_output
from tool_calling.evaluation.validation import validate_tool_output
from tool_calling.modeling.generation import (
    GENERATION_CONFIG,
    select_smoke_samples,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(
        description="批量运行 Train 候选样本，默认三类动作各一条。"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="运行全部 Train 候选样本。",
    )
    args = parser.parse_args()

    sample_path = PROJECT_ROOT / "data/generated/train_preview.jsonl"
    sample_text = sample_path.read_text(encoding="utf-8")
    samples = [
        json.loads(line)
        for line in sample_text.splitlines()
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
        raise ValueError("Train 候选样本文件为空。")

    sample_ids = [sample["id"] for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("候选样本编号重复。")

    # 加载模型前检查数据分区和参考目标，避免无效数据进入运行流程。
    for sample in samples:
        if sample["split"] != "train":
            raise ValueError("本脚本只允许读取 Train 候选样本。")

        errors = validate_tool_output(sample["expected"], tool_schemas)
        if errors:
            raise ValueError(
                f"参考目标不合法：{sample['id']}，{errors}"
            )

    if not args.full:
        samples = select_smoke_samples(samples)

    system_content = (
        rules.strip()
        + "\n\n可用工具定义：\n"
        + json.dumps(tools, ensure_ascii=False, indent=2)
    )
    model_path = PROJECT_ROOT / "models/qwen3_8b_4bit"

    # 记录实际配置与输入指纹，便于判断结果对应哪一版数据和提示词。
    run_info = {
        "mode": "train_development",
        "sample_selection": "all" if args.full else "first_per_action",
        "model_path": str(model_path),
        "adapter_path": None,
        "generation_config": GENERATION_CONFIG,
        "data_sha256": hashlib.sha256(
            sample_text.encode("utf-8")
        ).hexdigest(),
        "system_content_sha256": hashlib.sha256(
            system_content.encode("utf-8")
        ).hexdigest(),
    }

    print("本次样本数量：", len(samples))
    print("正在加载本地模型……", flush=True)

    # 模型只加载一次；每条请求单独构建对话，不携带上一条对话。
    model, tokenizer = load(str(model_path))

    output_dir = PROJECT_ROOT / "results/development"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )
    prefix = "train_batch" if args.full else "train_smoke"
    output_path = output_dir / f"{prefix}_{timestamp}.jsonl"

    # 独占创建结果文件，逐条保存并刷新，保留已完成的运行记录。
    with output_path.open("x", encoding="utf-8") as output_file:
        for index, sample in enumerate(samples, start=1):
            # 参考目标只用于生成后的评分，不进入模型输入。
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

            # 格式或接口不合格的输出也保存，不能跳过失败样本。
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
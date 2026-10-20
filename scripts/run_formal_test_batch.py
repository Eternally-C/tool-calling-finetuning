import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from tool_calling.data.frozen_test import load_frozen_test_inputs
from tool_calling.evaluation.scoring import score_tool_output
from tool_calling.paths import PROJECT_ROOT


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_run(variant):
    checked = load_frozen_test_inputs()
    plan = checked["plan"]
    model_path = (PROJECT_ROOT / plan["model"]["path"]).resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"本地模型目录不存在：{model_path}")

    checkpoint = None
    if variant == "baseline":
        if plan["baseline"]["adapter_path"] is not None:
            raise ValueError("锁定的 baseline 应当不加载 adapter。")
    elif variant == "step450":
        candidate = plan["candidate"]
        if candidate["label"] != variant:
            raise ValueError("候选名称与锁定方案不一致。")
        checkpoint = PROJECT_ROOT / candidate["checkpoint_path"]
        if file_sha256(checkpoint) != candidate["checkpoint_sha256"]:
            raise ValueError("候选 checkpoint 指纹不一致。")

        config_path = checkpoint.parent / "adapter_config.json"
        if file_sha256(config_path) != candidate["adapter_config_sha256"]:
            raise ValueError("候选训练配置指纹不一致。")
        config = json.loads(config_path.read_bytes())
        train_hash = file_sha256(
            PROJECT_ROOT / "data/processed/formal_sft/train.jsonl"
        )
        if (
            config["stage"] != "formal"
            or config["fine_tune_type"] != "lora"
            or config["train_data_sha256"] != train_hash
            or Path(config["model"]).resolve() != model_path
        ):
            raise ValueError("候选的训练来源或基座模型不一致。")
    else:
        raise ValueError("只允许 baseline 或 step450。")

    return checked, model_path, checkpoint


def main():
    parser = argparse.ArgumentParser(
        description="固定方案的最终 Test 入口，默认仅预检。"
    )
    parser.add_argument(
        "--variant", choices=("baseline", "step450"), required=True
    )
    parser.add_argument(
        "--run", action="store_true",
        help="实际评估全部 200 条 Test 样本。",
    )
    args = parser.parse_args()
    checked, model_path, checkpoint = prepare_run(args.variant)
    samples = checked["samples"]
    plan = checked["plan"]
    generation_config = plan["generation_config"]

    print("方案：", args.variant)
    print("样本数量：", len(samples))
    print("checkpoint：", checkpoint or "无，使用基线")
    print("生成设置：", generation_config)
    if not args.run:
        print("入口预检通过，未加载模型或执行推理。")
        return

    output_path = (
        PROJECT_ROOT / f"results/formal/test/{args.variant}.jsonl"
    )
    if output_path.exists():
        raise FileExistsError(f"Test 结果已存在，未覆盖：{output_path}")

    # 只有显式指定 --run 后，才导入模型框架并加载本地权重。
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler
    from tool_calling.modeling.loading import load_model_with_checkpoint

    print("正在加载本地模型……", flush=True)
    model, tokenizer, adapter_info = load_model_with_checkpoint(
        model_path, checkpoint
    )
    if checkpoint is not None:
        candidate = plan["candidate"]
        for key in ("checkpoint_sha256", "adapter_config_sha256"):
            if adapter_info[key] != candidate[key]:
                raise ValueError(f"实际加载后的 {key} 与锁定方案不一致。")
    elif adapter_info["checkpoint_path"] is not None:
        raise ValueError("baseline 意外加载了 checkpoint。")

    # 保存实际运行配置及代码版本，便于从原始输出复核结果。
    code_paths = [
        Path(__file__).resolve(),
        PROJECT_ROOT / "src/tool_calling/data/frozen_test.py",
        PROJECT_ROOT / "src/tool_calling/modeling/loading.py",
        PROJECT_ROOT / "src/tool_calling/evaluation/parsing.py",
        PROJECT_ROOT / "src/tool_calling/evaluation/validation.py",
        PROJECT_ROOT / "src/tool_calling/evaluation/scoring.py",
    ]
    run_info = {
        "stage": "formal",
        "mode": "test_baseline" if checkpoint is None else "test_lora",
        "variant": args.variant,
        "sample_selection": "all",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "model_repository": plan["model"]["repository"],
        "model_revision": plan["model"]["revision"],
        **adapter_info,
        **checked["freeze_info"],
        "generation_config": generation_config,
        "runtime_versions": {
            "python": platform.python_version(),
            "mlx": version("mlx"),
            "mlx_lm": version("mlx-lm"),
            "transformers": version("transformers"),
            "numpy": version("numpy"),
        },
        "code_sha256": {
            str(path.relative_to(PROJECT_ROOT)): file_sha256(path)
            for path in code_paths
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as output_file:
        for index, sample in enumerate(samples, start=1):
            # 每条独立对话；参考答案和来源信息不进入模型输入。
            messages = [
                {"role": "system", "content": checked["system_content"]},
                {"role": "user", "content": sample["query"]},
            ]
            prompt_tokens = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                enable_thinking=generation_config["enable_thinking"],
            )
            print(
                f"[{index}/{len(samples)}] {sample['id']}",
                flush=True,
            )
            response = generate(
                model,
                tokenizer,
                prompt=prompt_tokens,
                max_tokens=generation_config["max_tokens"],
                sampler=make_sampler(
                    temp=generation_config["temperature"]
                ),
                verbose=False,
            )
            score = score_tool_output(
                response, sample["expected"], checked["tool_schemas"]
            )
            # 答错或格式失败也写入记录，不删除、不跳过。
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

    print("最终 Test 推理完成。")
    print("结果文件：", output_path)


if __name__ == "__main__":
    main()
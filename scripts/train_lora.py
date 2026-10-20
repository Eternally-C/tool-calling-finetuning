import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from mlx_lm import load
from mlx_lm.lora import CONFIG_DEFAULTS, train_model
from mlx_lm.tuner.callbacks import TrainingCallback
from mlx_lm.tuner.datasets import ChatDataset

from tool_calling.modeling.tokenization import set_non_thinking_default


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class MetricsCallback(TrainingCallback):
    def __init__(self, output_file):
        self.output_file = output_file

    def on_train_loss_report(self, train_info):
        # 每条记录对应一个报告区间，loss 是该区间的平均值。
        record = {"split": "train", **train_info}
        self.output_file.write(
            json.dumps(
                record,
                ensure_ascii=False,
                allow_nan=False,
            ) + "\n"
        )
        self.output_file.flush()

        print(
            f"已完成 {train_info['iteration']} 步："
            f"平均 loss={train_info['train_loss']:.4f}，"
            f"MLX 峰值内存={train_info['peak_memory']:.2f} GB",
            flush=True,
        )


def main():
    data_dir = PROJECT_ROOT / "data/processed/pilot_sft"
    if (data_dir / "test.jsonl").exists():
        raise ValueError("训练数据目录中不能包含 test.jsonl。")

    # 参数更新只读取 Train，Validation 在训练后单独生成和评分。
    train_path = data_dir / "train.jsonl"
    train_bytes = train_path.read_bytes()
    records = [
        json.loads(line)
        for line in train_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("Train 数据为空。")

    sample_ids = [record["metadata"]["id"] for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Train 样本编号重复。")

    for record in records:
        if record["metadata"]["split"] != "train":
            raise ValueError("训练输入中存在非 Train 样本。")
        roles = [message["role"] for message in record["messages"]]
        if roles != ["system", "user", "assistant"]:
            raise ValueError(
                f"训练消息结构异常：{record['metadata']['id']}"
            )

    model_path = PROJECT_ROOT / "models/qwen3_8b_4bit"
    model_config = json.loads(
        (model_path / "config.json").read_text(encoding="utf-8")
    )
    if model_config.get("quantization", {}).get("bits") != 4:
        raise ValueError("本轮实验要求使用已有的 4-bit 基座模型。")

    # 当前固定 batch_size=1、梯度累积=1，所以每轮步数等于样本数。
    num_epochs = 3
    steps_per_epoch = len(records)
    total_steps = num_epochs * steps_per_epoch

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output_dir = PROJECT_ROOT / "adapters/pilot" / f"run_{run_id}"
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在：{output_dir}")

    config = deepcopy(CONFIG_DEFAULTS)
    config.update({
        "model": str(model_path),
        "data": str(data_dir),
        "train": True,
        "test": False,
        "seed": 42,
        "fine_tune_type": "lora",
        "num_layers": 16,
        "batch_size": 1,
        "iters": total_steps,
        "learning_rate": 1e-5,
        "steps_per_report": max(1, steps_per_epoch // 4),
        "save_every": steps_per_epoch,
        "max_seq_length": 2048,
        "grad_checkpoint": True,
        "grad_accumulation_steps": 1,
        "mask_prompt": True,
        "resume_adapter_file": None,
        "lora_parameters": {
            "rank": 8,
            "dropout": 0.0,
            "scale": 20.0,
        },
        "adapter_path": str(output_dir),
        "experiment_stage": "pilot",
        "num_epochs": num_epochs,
        "validation_during_training": False,
        "enable_thinking": False,
        "mlx_lm_version": version("mlx-lm"),
        "mlx_version": version("mlx"),
        "train_data_sha256": hashlib.sha256(train_bytes).hexdigest(),
    })
    args = SimpleNamespace(**config)

    print("Train 样本数：", len(records))
    print("训练轮数：", num_epochs)
    print("每轮更新步数：", steps_per_epoch)
    print("总更新步数：", total_steps)
    print("实验输出目录：", output_dir)
    print("正在加载本地基座模型……", flush=True)

    np.random.seed(args.seed)
    model, tokenizer = load(
        str(model_path),
        tokenizer_config={
            "local_files_only": True,
            "trust_remote_code": False,
        },
        trust_remote_code=False,
    )
    tokenizer = set_non_thinking_default(tokenizer)
    train_set = ChatDataset(records, tokenizer, mask_prompt=True)

    # 在实际训练进程中检查完整序列，防止截断答案或错置监督边界。
    for record in records:
        tokens, offset = train_set.process(record)
        prompt_tokens = tokenizer.apply_chat_template(
            record["messages"][:-1],
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=False,
        )
        if (
            offset != len(prompt_tokens)
            or tokens[:offset] != prompt_tokens
            or not 0 < offset < len(tokens)
        ):
            raise ValueError(
                f"监督边界异常：{record['metadata']['id']}"
            )
        if len(tokens) > args.max_seq_length:
            raise ValueError(
                f"训练样本超长：{record['metadata']['id']}"
            )

    output_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = output_dir / "training_metrics.jsonl"

    # 独占创建日志并逐次刷新，训练中断时也保留已完成的记录。
    with metrics_path.open("x", encoding="utf-8") as output_file:
        train_model(
            args=args,
            model=model,
            train_set=train_set,
            valid_set=[],
            training_callback=MetricsCallback(output_file),
        )

    print("本轮训练完成。")
    print("训练指标日志：", metrics_path)
    print("checkpoint 目录：", output_dir)


if __name__ == "__main__":
    main()
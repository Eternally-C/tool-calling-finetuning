import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import version
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.lora import CONFIG_DEFAULTS, train_model
from mlx_lm.tuner.callbacks import TrainingCallback
from mlx_lm.tuner.datasets import ChatDataset

from tool_calling.modeling.tokenization import set_non_thinking_default
from tool_calling.paths import PROJECT_ROOT


class ProbeCallback(TrainingCallback):
    def on_train_loss_report(self, train_info):
        # 这是 MLX 统计的峰值分配内存，不是整台电脑的内存占用。
        print(
            f"试跑第 {train_info['iteration']} 步："
            f"loss={train_info['train_loss']:.4f}，"
            f"MLX 峰值内存={train_info['peak_memory']:.2f} GB",
            flush=True,
        )


def main():
    data_dir = PROJECT_ROOT / "data/processed/pilot_sft"
    if (data_dir / "test.jsonl").exists():
        raise ValueError("试训目录中不能包含 test.jsonl。")

    # 本次资源试跑只使用 Train，不进行验证集效果评估。
    train_path = data_dir / "train.jsonl"
    train_bytes = train_path.read_bytes()
    records = [
        json.loads(line)
        for line in train_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("Train 数据为空。")
    if any(record["metadata"]["split"] != "train" for record in records):
        raise ValueError("训练输入中存在非 Train 样本。")
    sample_ids = [record["metadata"]["id"] for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("训练样本编号重复。")

    model_path = PROJECT_ROOT / "models/qwen3_8b_4bit"
    model_config = json.loads(
        (model_path / "config.json").read_text(encoding="utf-8")
    )
    if model_config.get("quantization", {}).get("bits") != 4:
        raise ValueError("本次试跑要求使用已有的 4-bit 基座模型。")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    adapter_path = PROJECT_ROOT / "adapters/probes" / f"probe_{run_id}"
    if adapter_path.exists():
        raise FileExistsError(f"输出目录已存在：{adapter_path}")

    # 显式覆盖本次试跑参数；框架会将配置保存到输出目录。
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
        "iters": 10,
        "learning_rate": 1e-5,
        "steps_per_report": 1,
        "steps_per_eval": 10,
        "save_every": 10,
        "max_seq_length": 2048,
        "grad_checkpoint": True,
        "grad_accumulation_steps": 1,
        "mask_prompt": True,
        "lora_parameters": {
            "rank": 8,
            "dropout": 0.0,
            "scale": 20.0,
        },
        "adapter_path": str(adapter_path),
        "probe_only": True,
        "enable_thinking": False,
        "mlx_lm_version": version("mlx-lm"),
        "mlx_version": version("mlx"),
        "train_data_sha256": hashlib.sha256(train_bytes).hexdigest(),
    })
    args = SimpleNamespace(**config)

    print("Train 样本数：", len(records))
    print("训练步数：", args.iters)
    print("临时 adapter 输出目录：", adapter_path)
    print("正在加载本地模型……", flush=True)

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

    # 使用训练进程实际拿到的 tokenizer，再检查监督边界和完整长度。
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
                f"训练监督边界异常：{record['metadata']['id']}"
            )
        if len(tokens) > args.max_seq_length:
            raise ValueError(
                f"训练样本超长：{record['metadata']['id']}"
            )

    print("训练输入检查通过，开始 10 步试跑。", flush=True)
    # 空数据集表示本次跳过验证，兼容框架的数据集包装逻辑。
    train_model(
        args=args,
        model=model,
        train_set=train_set,
        valid_set=[],
        training_callback=ProbeCallback(),
    )

    print("10 步训练试跑完成。")
    print("最终 MLX 峰值内存：", round(mx.get_peak_memory() / 1e9, 2), "GB")
    print("临时 adapter 输出目录：", adapter_path)


if __name__ == "__main__":
    main()
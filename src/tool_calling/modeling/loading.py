import hashlib
import json
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers


def load_model_with_checkpoint(model_path, checkpoint_path=None):
    model_path = Path(model_path).resolve()
    checkpoint = None
    config = None
    adapter_info = {
        "adapter_path": None,
        "checkpoint_path": None,
        "checkpoint_sha256": None,
        "adapter_config_sha256": None,
        "train_data_sha256": None,
    }

    if checkpoint_path is not None:
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.is_file() or checkpoint.suffix != ".safetensors":
            raise ValueError(f"checkpoint 文件无效：{checkpoint}")

        config_path = checkpoint.parent / "adapter_config.json"
        config_bytes = config_path.read_bytes()
        config = json.loads(config_bytes.decode("utf-8"))

        if config["fine_tune_type"] != "lora":
            raise ValueError("本项目加载模块目前只支持 LoRA checkpoint。")
        if Path(config["model"]).resolve() != model_path:
            raise ValueError("checkpoint 记录的基座模型路径不一致。")

        adapter_info.update({
            "adapter_path": str(checkpoint.parent),
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(
                checkpoint.read_bytes()
            ).hexdigest(),
            "adapter_config_sha256": hashlib.sha256(
                config_bytes
            ).hexdigest(),
            "train_data_sha256": config["train_data_sha256"],
        })

    model, tokenizer = load(
        str(model_path),
        tokenizer_config={
            "local_files_only": True,
            "trust_remote_code": False,
        },
        trust_remote_code=False,
    )

    if checkpoint is not None:
        if not 1 <= config["num_layers"] <= len(model.layers):
            raise ValueError("checkpoint 的 LoRA 层数设置无效。")

        # 按训练配置恢复 LoRA 结构，再载入明确指定的 checkpoint。
        model.freeze()
        linear_to_lora_layers(
            model,
            config["num_layers"],
            config["lora_parameters"],
        )
        weights = mx.load(str(checkpoint))

        # 要求全部 LoRA 参数齐全，且名称与形状完全匹配。
        expected_shapes = {
            name: value.shape
            for name, value in tree_flatten(model.parameters())
            if name.rsplit(".", 1)[-1] in ("lora_a", "lora_b")
        }
        if set(weights) != set(expected_shapes):
            raise ValueError("checkpoint 的 LoRA 参数集合不完整或不匹配。")
        for name, value in weights.items():
            if value.shape != expected_shapes[name]:
                raise ValueError(f"LoRA 参数形状不匹配：{name}")

        model.load_weights(list(weights.items()), strict=False)
        mx.eval(model.parameters())

    model.eval()
    return model, tokenizer, adapter_info
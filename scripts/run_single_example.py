import argparse
import json
from pathlib import Path

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from tool_calling.evaluation.scoring import score_tool_output


# 根据脚本位置确定项目目录，避免依赖终端当前所在目录。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
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
    examples = json.loads(
        (PROJECT_ROOT / "data/manual/train_examples.json").read_text(
            encoding="utf-8"
        )
    )
    # 通过命令行指定开发样例编号，默认仍使用正常库存查询。
    parser = argparse.ArgumentParser(
        description="运行单条工具调用开发样例"
    )
    parser.add_argument(
        "--example-id",
        default="train_002",
        choices=[item["id"] for item in examples],
        help="需要检查的样例编号",
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=None,
        help="LoRA adapter 目录；不提供时运行基线模型",
    )
    args = parser.parse_args()

    example = next(
        item for item in examples if item["id"] == args.example_id
    )

    # 将工具定义作为提示词内容提供，采用本项目的统一 JSON 输出协议。
    system_content = (
        rules.strip()
        + "\n\n可用工具定义：\n"
        + json.dumps(tools, ensure_ascii=False, indent=2)
    )

    # 输入只包含规则、工具定义和用户请求；参考答案不进入模型输入。
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": example["query"]},
    ]

    print("用户请求：", example["query"])
    print("正在加载本地模型……", flush=True)

    model_path = PROJECT_ROOT / "models/qwen3_8b_4bit"
    adapter_path = args.adapter_path

    # 相对路径统一以项目根目录为起点。
    if adapter_path is not None:
        adapter_path = adapter_path.expanduser()
        if not adapter_path.is_absolute():
            adapter_path = PROJECT_ROOT / adapter_path
        adapter_path = adapter_path.resolve()

        for filename in ("adapter_config.json", "adapters.safetensors"):
            if not (adapter_path / filename).is_file():
                raise FileNotFoundError(
                    f"adapter 缺少必要文件：{adapter_path / filename}"
                )

        print("本次加载的 adapter：", adapter_path)
    else:
        print("本次运行：基线模型")

    # 在同一基座上加载指定 adapter，使用本地 tokenizer。
    model, tokenizer = load(
        str(model_path),
        adapter_path=str(adapter_path) if adapter_path is not None else None,
        tokenizer_config={
            "local_files_only": True,
            "trust_remote_code": False,
        },
        trust_remote_code=False,
    )

    # 使用模型的对话模板，并关闭思考输出模式。
    prompt_tokens = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    print("输入 token 数：", len(prompt_tokens))
    print("正在生成模型输出……", flush=True)

    # 温度为 0 时采用贪心解码，减少随机采样带来的输出变化。
    response = generate(
        model,
        tokenizer,
        prompt=prompt_tokens,
        max_tokens=256,
        sampler=make_sampler(temp=0.0),
        verbose=False,
    )

    print("模型输出：", response)
    print(
        "参考目标：",
        json.dumps(example["expected"], ensure_ascii=False),
    )

    # 统一评分，确保单条运行与后续批量检查使用相同口径。
    result = score_tool_output(
        response, example["expected"], tool_schemas
    )
    print("评分结果：")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
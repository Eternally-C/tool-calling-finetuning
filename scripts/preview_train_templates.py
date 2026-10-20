import argparse
import json
from pathlib import Path

from tool_calling.data.templates import fill_placeholders


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    templates = json.loads(
        (PROJECT_ROOT / "data/manual/train_templates.json").read_text(
            encoding="utf-8"
        )
    )

    parser = argparse.ArgumentParser(
        description="预览训练模板生成的请求与参考目标。"
    )
    parser.add_argument(
        "--category",
        choices=["electronics", "clothing", "home"],
        default="clothing",
        help="本次预览使用的商品分类，默认 clothing。",
    )
    parser.add_argument(
        "--product-id",
        default="P9001",
        help="本次预览使用的商品编号，默认 P9001。",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=10,
        help="本次预览使用的库存阈值，必须为正整数，默认 10。",
    )
    args = parser.parse_args()

    # 检查用于填充正常调用模板的参数，避免生成不合法的参考目标。
    if not args.product_id.strip():
        parser.error("商品编号不能为空。")
    if args.threshold < 1:
        parser.error("库存阈值必须是大于零的整数。")

    # 中文请求与英文枚举共用同一映射，避免分类名称和目标参数不一致。
    category_names = {
        "electronics": "电子产品",
        "clothing": "服装",
        "home": "家居用品",
    }

    # 请求与参考目标共用这些变量，保证编号、分类和阈值对应一致。
    variables = {
        "product_id": args.product_id,
        "order_id": "O9001",
        "refund_id": "R9001",
        "category_zh": category_names[args.category],
        "category": args.category,
        "start_date": "2026-09-01",
        "end_date": "2026-09-30",
        "threshold": args.threshold,
    }
    examples = []

    for template in templates:
        if template["split"] != "train":
            raise ValueError("训练模板文件中出现了非 train 模板。")

        # 同一组变量同时填充请求和参考目标，避免参数不一致。
        example = {
            # 当前每个模板只生成一条预览，使用模板编号保证唯一。
            "id": f"preview_{template['template_id']}",
            "template_id": template["template_id"],
            "family_id": template["family_id"],
            "split": template["split"],
            "case_type": template["case_type"],
            "query": fill_placeholders(
                template["query_template"], variables
            ),
            "expected": fill_placeholders(
                template["expected_template"], variables
            ),
        }
        examples.append(example)

    # 输出预览，保留来源信息，方便核查和后续错误追踪。
    print("预览商品分类：", args.category, "/", variables["category_zh"])
    print("预览样本数量：", len(examples))
    print(json.dumps(examples, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
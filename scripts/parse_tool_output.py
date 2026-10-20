import argparse
import json

from tool_calling.evaluation.parsing import parse_tool_output


def main():
    parser = argparse.ArgumentParser(
        description="检查工具调用原始输出的 JSON 格式。"
    )
    parser.add_argument("raw_output", help="需要检查的原始 JSON 字符串。")
    args = parser.parse_args()

    try:
        output = parse_tool_output(args.raw_output)
    except ValueError as error:
        print("JSON 对象解析：失败，", error)
        raise SystemExit(1)

    print("JSON 对象解析：通过")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
import hashlib
import json
import random
from collections import Counter, defaultdict

from tool_calling.evaluation.validation import validate_tool_output
from tool_calling.paths import PROJECT_ROOT


DATA_DIR = PROJECT_ROOT / "data/generated/formal"
INPUT_PATH = DATA_DIR / "train_candidates.jsonl"
OUTPUT_PATH = DATA_DIR / "train_selected.jsonl"
PLAN_PATH = PROJECT_ROOT / "configs/formal_data_plan.json"
SEED = 42


def select_examples():
    candidate_bytes = INPUT_PATH.read_bytes()
    plan_bytes = PLAN_PATH.read_bytes()
    source_bytes = {
        "templates_sha256": (
            PROJECT_ROOT / "data/manual/formal/train_templates.json"
        ).read_bytes(),
        "parameter_sets_sha256": (
            DATA_DIR / "train_parameter_sets.json"
        ).read_bytes(),
        "tools_sha256": (
            PROJECT_ROOT / "configs/tools.json"
        ).read_bytes(),
    }
    expected_source = {
        "stage": "formal",
        **{
            key: hashlib.sha256(value).hexdigest()
            for key, value in source_bytes.items()
        },
    }
    templates = {
        item["template_id"]: item
        for item in json.loads(source_bytes["templates_sha256"])
    }
    parameters = {
        item["parameter_set_id"]: item
        for item in json.loads(source_bytes["parameter_sets_sha256"])
    }
    tool_schemas = {
        tool["function"]["name"]: tool["function"]["parameters"]
        for tool in json.loads(source_bytes["tools_sha256"])
    }
    targets = json.loads(plan_bytes)["target_action_counts"]["train"]
    actions = ("call", "clarify", "unsupported")
    if set(targets) != set(actions) or any(
        type(value) is not int or value < 1 for value in targets.values()
    ):
        raise ValueError("动作目标数量必须为三个类别的正整数。")

    examples = [
        json.loads(line)
        for line in candidate_bytes.decode("utf-8").splitlines()
    ]
    for field in ("id", "query"):
        values = [example[field] for example in examples]
        if len(values) != len(set(values)):
            raise ValueError(f"候选中的 {field} 存在重复。")
    if {item["template_id"] for item in examples} != set(templates):
        raise ValueError("候选未覆盖全部模板。")

    def coverage_key(example):
        template = templates[example["template_id"]]
        parameter = parameters[example["parameter_set_id"]]
        query_template = template["query_template"]
        # 只按请求实际使用的枚举分组，忽略未进入请求的参数。
        category = (
            parameter["category"]
            if "{category_zh}" in query_template else None
        )
        metric = (
            parameter["metric"]
            if "{metric_zh}" in query_template else None
        )
        expected = example["expected"]
        same_day = None
        if expected.get("name") == "get_sales_summary":
            arguments = expected["arguments"]
            # 同一天与普通日期区间分别保留，避免丢失合法边界。
            same_day = arguments["start_date"] == arguments["end_date"]
        return example["template_id"], category, metric, same_day

    groups = defaultdict(list)
    for example in examples:
        template = templates[example["template_id"]]
        parameter = parameters[example["parameter_set_id"]]
        if any(item["split"] != "train" for item in (
            example, template, parameter
        )):
            raise ValueError("抽样来源出现了其他分区。")
        if example["source_info"] != expected_source:
            raise ValueError("候选来源已变化，请先核查数据版本。")
        errors = validate_tool_output(example["expected"], tool_schemas)
        if errors:
            raise ValueError(f"{example['id']}：{errors}")
        groups[coverage_key(example)].append(example)

    rng = random.Random(SEED)
    # 每个覆盖组先保留一条；只有一条的边界样本必然被保留。
    mandatory = [rng.choice(group) for group in groups.values()]
    mandatory_ids = {example["id"] for example in mandatory}
    selected = list(mandatory)

    for action in actions:
        kept_count = sum(
            item["expected"]["action"] == action for item in mandatory
        )
        needed = targets[action] - kept_count
        pool = [
            item for item in examples
            if item["expected"]["action"] == action
            and item["id"] not in mandatory_ids
        ]
        if not 0 <= needed <= len(pool):
            raise ValueError(f"{action} 无法满足数量与覆盖要求。")
        # 不放回抽样，保留项计入目标数量，不额外复制样本。
        selected.extend(rng.sample(pool, needed))

    if Counter(item["expected"]["action"] for item in selected) != targets:
        raise ValueError("最终动作数量不符合计划。")
    if {coverage_key(item) for item in selected} != set(groups):
        raise ValueError("抽样后存在覆盖缺失。")
    if {
        item["expected"]["name"] for item in selected
        if item["expected"]["action"] == "call"
    } != set(tool_schemas):
        raise ValueError("抽样后未覆盖全部工具。")

    rng.shuffle(selected)
    selection_info = {
        "method": "coverage_then_action_sampling_v1",
        "seed": SEED,
        "candidates_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "target_action_counts": targets,
        "mandatory_count": len(mandatory),
    }
    # 来源和抽样信息用于追溯，后续不放入模型的对话输入。
    selected = [
        {
            **item,
            "selection_info": {
                **selection_info,
                "mandatory": item["id"] in mandatory_ids,
            },
        }
        for item in selected
    ]
    return selected, Counter(
        item["expected"]["action"] for item in mandatory
    )


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"抽样文件已存在，未覆盖：{OUTPUT_PATH}")

    selected, mandatory_counts = select_examples()
    # 全部检查通过后才创建新文件，原候选池保持不变。
    with OUTPUT_PATH.open("x", encoding="utf-8") as output_file:
        for example in selected:
            output_file.write(
                json.dumps(example, ensure_ascii=False) + "\n"
            )

    print("优先保留数量：", sum(mandatory_counts.values()))
    print("优先保留动作分布：", dict(mandatory_counts))
    print("最终样本数量：", len(selected))
    print("最终动作分布：", dict(Counter(
        item["expected"]["action"] for item in selected
    )))
    print("模板覆盖数量：", len({
        item["template_id"] for item in selected
    }))
    print("call 工具分布：", dict(Counter(
        item["expected"]["name"] for item in selected
        if item["expected"]["action"] == "call"
    )))
    print("抽样文件：", OUTPUT_PATH)


if __name__ == "__main__":
    main()
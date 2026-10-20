import hashlib
import json
import random
from collections import Counter, defaultdict

from preview_formal_test_batch import (
    PARAMETER_PATH,
    TEMPLATE_PATH,
    build_candidates,
)
from tool_calling.paths import PROJECT_ROOT


DATA_DIR = PROJECT_ROOT / "data/generated/formal"
INPUT_PATH = DATA_DIR / "test_candidates.jsonl"
OUTPUT_PATH = DATA_DIR / "test_selected.jsonl"
PLAN_PATH = PROJECT_ROOT / "configs/final_evaluation_plan.json"


def select_examples():
    candidate_bytes = INPUT_PATH.read_bytes()
    plan_bytes = PLAN_PATH.read_bytes()
    plan = json.loads(plan_bytes)
    policy_path = PROJECT_ROOT / plan["test_policy"]["split_policy_file"]
    policy_bytes = policy_path.read_bytes()
    policy = json.loads(policy_bytes)
    seed = policy["shuffle_seed"]
    if type(seed) is not int:
        raise ValueError("随机种子必须为整数。")
    if policy["evaluation"]["allow_test_tuning"] is not False:
        raise ValueError("数据分区策略不允许开启 Test 调参。")

    examples = [
        json.loads(line)
        for line in candidate_bytes.decode("utf-8").splitlines()
    ]
    # 纯构造函数不会写文件：逐条核对请求、参考目标及来源指纹。
    rebuilt, _ = build_candidates()
    if examples != rebuilt:
        raise ValueError("候选文件与当前来源不一致，请先核查数据版本。")

    templates = {
        item["template_id"]: item
        for item in json.loads(TEMPLATE_PATH.read_bytes())
    }
    parameters = {
        item["parameter_set_id"]: item
        for item in json.loads(PARAMETER_PATH.read_bytes())
    }
    targets = plan["test_policy"]["target_action_counts"]
    actions = ("call", "clarify", "unsupported")
    if set(targets) != set(actions) or any(
        type(value) is not int or value < 1 for value in targets.values()
    ):
        raise ValueError("动作目标数量必须为三个类别的正整数。")
    if sum(targets.values()) != 200:
        raise ValueError("当前锁定的 Test 总量应为 200 条。")

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
        same_day = None
        expected = example["expected"]
        if expected.get("name") == "get_sales_summary":
            arguments = expected["arguments"]
            same_day = arguments["start_date"] == arguments["end_date"]
        return example["template_id"], category, metric, same_day

    groups = defaultdict(list)
    for example in examples:
        groups[coverage_key(example)].append(example)

    rng = random.Random(seed)
    # 每个覆盖组先保留一条；只有一条的同日边界也会被保留。
    mandatory = [rng.choice(group) for group in groups.values()]
    mandatory_ids = {item["id"] for item in mandatory}
    mandatory_counts = Counter(
        item["expected"]["action"] for item in mandatory
    )
    selected = list(mandatory)

    for action in actions:
        needed = targets[action] - mandatory_counts[action]
        pool = [
            item for item in examples
            if item["expected"]["action"] == action
            and item["id"] not in mandatory_ids
        ]
        if not 0 <= needed <= len(pool):
            raise ValueError(f"{action} 无法同时满足数量与覆盖要求。")
        # 必选项计入配额，剩余部分不放回抽样。
        selected.extend(rng.sample(pool, needed))

    if Counter(item["expected"]["action"] for item in selected) != targets:
        raise ValueError("最终动作数量不符合锁定方案。")
    for field in ("id", "query"):
        if len({item[field] for item in selected}) != len(selected):
            raise ValueError(f"选样后出现重复的 {field}。")
    if {coverage_key(item) for item in selected} != set(groups):
        raise ValueError("选样后存在覆盖缺失。")

    tools = json.loads((PROJECT_ROOT / "configs/tools.json").read_bytes())
    selected_tools = {
        item["expected"]["name"] for item in selected
        if item["expected"]["action"] == "call"
    }
    if selected_tools != {tool["function"]["name"] for tool in tools}:
        raise ValueError("选样后未覆盖全部工具。")

    rng.shuffle(selected)
    selection_info = {
        "method": "coverage_then_action_sampling_v1",
        "seed": seed,
        "candidates_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "split_policy_sha256": hashlib.sha256(policy_bytes).hexdigest(),
        "target_action_counts": targets,
        "mandatory_count": len(mandatory),
    }
    # 这些信息用于追溯，不放入模型的对话输入。
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
    return selected, mandatory_counts


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"选样文件已存在，未覆盖：{OUTPUT_PATH}")

    selected, mandatory_counts = select_examples()
    # 全部检查通过后才创建文件；候选池保持不变。
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
    print("选样文件：", OUTPUT_PATH)


if __name__ == "__main__":
    main()
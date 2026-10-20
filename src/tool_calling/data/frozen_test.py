import hashlib
import json
from collections import Counter

from tool_calling.evaluation.validation import validate_tool_output
from tool_calling.paths import PROJECT_ROOT


MANIFEST_PATH = PROJECT_ROOT / "configs/test_data_manifest.json"
EXPECTED_MANIFEST_SHA256 = (
    "54144c46ccf77ddc0f2a021f18b67ba16f2a0c70e1062ce6d85c8d2cd0085cfe"
)


def load_frozen_test_inputs():
    """校验冻结清单，返回 Test 输入；不加载模型或执行推理。"""
    manifest_bytes = MANIFEST_PATH.read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_hash != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Test 冻结清单发生变化，请先核查，不能继续推理。")
    manifest = json.loads(manifest_bytes)

    if (
        manifest["status"] != "frozen_before_inference"
        or manifest["policy"]["verify_hashes_before_inference"] is not True
        or manifest["policy"]["same_test_data_for_baseline_and_candidate"]
        is not True
        or manifest["policy"]["allow_test_tuning"] is not False
    ):
        raise ValueError("Test 冻结状态或评估策略不符合要求。")

    # 从已核对指纹的同一份字节解析输入，避免校验后再次读取不同版本。
    snapshots = {}
    for relative_path, expected_hash in manifest["files_sha256"].items():
        content = (PROJECT_ROOT / relative_path).read_bytes()
        if hashlib.sha256(content).hexdigest() != expected_hash:
            raise ValueError(f"冻结文件指纹不一致：{relative_path}")
        snapshots[relative_path] = content

    plan_path = "configs/final_evaluation_plan.json"
    plan = json.loads(snapshots[plan_path])
    if (
        plan["status"] != "locked_before_test"
        or plan["test_policy"]["freeze_test_data_before_inference"]
        is not True
        or plan["test_policy"]["allow_test_tuning"] is not False
    ):
        raise ValueError("最终评估方案没有保持锁定。")

    data_bytes = snapshots[manifest["selected_file"]]
    samples = [
        json.loads(line)
        for line in data_bytes.decode("utf-8").splitlines()
    ]
    tools = json.loads(snapshots[plan["frozen_inputs"]["tools_file"]])
    tool_schemas = {
        tool["function"]["name"]: tool["function"]["parameters"]
        for tool in tools
    }
    for sample in samples:
        if sample["split"] != "test":
            raise ValueError(f"Test 数据中出现其他分区：{sample['id']}")
        errors = validate_tool_output(sample["expected"], tool_schemas)
        if errors:
            raise ValueError(f"{sample['id']}：{errors}")

    counts = Counter(sample["expected"]["action"] for sample in samples)
    if (
        len(samples) != manifest["sample_count"]
        or counts != manifest["action_counts"]
        or counts != plan["test_policy"]["target_action_counts"]
    ):
        raise ValueError("Test 样本数量或动作分布不符合冻结记录。")
    for field in ("id", "query"):
        if len({sample[field] for sample in samples}) != len(samples):
            raise ValueError(f"Test 数据存在重复的 {field}。")

    coverage = {
        "template_count": len({s["template_id"] for s in samples}),
        "family_count": len({s["family_id"] for s in samples}),
        "tool_count": len({
            s["expected"]["name"] for s in samples
            if s["expected"]["action"] == "call"
        }),
        "same_day_sales_call_count": sum(
            s["expected"].get("name") == "get_sales_summary"
            and s["expected"]["arguments"]["start_date"]
            == s["expected"]["arguments"]["end_date"]
            for s in samples
        ),
    }
    if coverage != manifest["coverage"]:
        raise ValueError("Test 覆盖情况与冻结记录不一致。")
    if sum(
        s["selection_info"]["mandatory"] for s in samples
    ) != manifest["sampling"]["mandatory_count"]:
        raise ValueError("Test 必选样本数量与冻结记录不一致。")

    # 保持与 Validation 完全相同的工具定义和系统提示词构造方式。
    rules = snapshots[
        plan["frozen_inputs"]["system_prompt_file"]
    ].decode("utf-8")
    system_content = (
        rules.strip()
        + "\n\n可用工具定义：\n"
        + json.dumps(tools, ensure_ascii=False, indent=2)
    )
    system_hash = hashlib.sha256(system_content.encode("utf-8")).hexdigest()
    if system_hash != plan["frozen_inputs"]["system_content_sha256"]:
        raise ValueError("最终系统提示词与锁定方案不一致。")

    return {
        "manifest": manifest,
        "plan": plan,
        "samples": samples,
        "tools": tools,
        "tool_schemas": tool_schemas,
        "system_content": system_content,
        "freeze_info": {
            "manifest_sha256": manifest_hash,
            "plan_sha256": hashlib.sha256(snapshots[plan_path]).hexdigest(),
            "data_sha256": hashlib.sha256(data_bytes).hexdigest(),
            "system_content_sha256": system_hash,
        },
    }


def main():
    checked = load_frozen_test_inputs()
    print("冻结文件指纹通过：", len(
        checked["manifest"]["files_sha256"]
    ))
    print("Test 样本数量：", len(checked["samples"]))
    print("动作分布：", dict(Counter(
        sample["expected"]["action"] for sample in checked["samples"]
    )))
    print("覆盖记录和系统提示词指纹通过。")
    print("预检完成，未加载模型或执行推理。")


if __name__ == "__main__":
    main()
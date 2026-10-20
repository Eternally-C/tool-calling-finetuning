from datetime import date


def validate_date_range(parameter_set):
    """检查参数组的日期格式、日历有效性与起止顺序。"""
    set_id = parameter_set["parameter_set_id"]
    parsed_dates = {}

    for field in ("start_date", "end_date"):
        value = parameter_set[field]

        if not isinstance(value, str):
            raise ValueError(f"{set_id}：{field} 必须是日期字符串。")

        # 转成日期对象，检查日期是否真实存在，例如拒绝 2 月 30 日。
        try:
            parsed = date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(
                f"{set_id}：{field} 必须是有效的 YYYY-MM-DD 日期：{value}"
            ) from exc

        # 统一使用 YYYY-MM-DD，拒绝其他可解析但不符合项目约定的写法。
        if parsed.isoformat() != value:
            raise ValueError(
                f"{set_id}：{field} 必须使用 YYYY-MM-DD 格式：{value}"
            )

        parsed_dates[field] = parsed

    # 正常参数组允许查询同一天；倒置请求由错误案例模板主动构造。
    if parsed_dates["end_date"] < parsed_dates["start_date"]:
        raise ValueError(f"{set_id}：结束日期不能早于开始日期。")
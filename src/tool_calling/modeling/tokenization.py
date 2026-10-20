from functools import partial


def set_non_thinking_default(tokenizer):
    """在当前进程中，将聊天模板的 thinking 默认值设为 False。"""
    # 保留原有模板调用逻辑，只补充默认参数。
    tokenizer.apply_chat_template = partial(
        tokenizer.apply_chat_template,
        enable_thinking=False,
    )
    return tokenizer
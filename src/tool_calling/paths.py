from pathlib import Path


# 本文件位于 src/tool_calling/paths.py，统一从这里定位项目根目录。
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
ADAPTER_DIR = PROJECT_ROOT / "adapters"
RESULT_DIR = PROJECT_ROOT / "results"


def main():
    print("路径模块位置：", Path(__file__).resolve())
    print("项目根目录：", PROJECT_ROOT)

    for label, path in (
        ("配置目录", CONFIG_DIR),
        ("数据目录", DATA_DIR),
        ("模型目录", MODEL_DIR),
        ("adapter 目录", ADAPTER_DIR),
        ("结果目录", RESULT_DIR),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"{label}不存在：{path}")
        print(f"{label}：", path)

    print("公共包导入与项目目录定位检查通过。")


if __name__ == "__main__":
    main()
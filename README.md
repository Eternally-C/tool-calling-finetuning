# Parameter-Efficient Fine-Tuning and Reliability Evaluation for Tool Calling

面向中文电商查询请求，使用本地 Qwen3-8B 4-bit 量化基座与 MLX-LM LoRA，
评估工具选择、参数生成、澄清决策和不支持操作的识别。

## 任务范围

模型输出采用自定义 JSON 协议：

- call：选择工具并生成参数。
- clarify：要求补充、修正或澄清必要信息。
- unsupported：说明当前工具不支持的操作。

项目定义了 8 个查询工具，覆盖商品信息、库存、订单、物流、
已有退款申请状态、退货规则、销售统计和低库存商品筛选。

本项目评估调用决定，不连接业务数据库、不执行真实工具，
也未实现多轮 Agent。

## 方法

- 基座模型：`mlx-community/Qwen3-8B-4bit`。
- 模型 revision：`545dc4251c05440727734bcd94334791f6ab0192`。
- 正式训练：600 条 Train 样本，1 个 epoch，共 600 次更新。
- LoRA：最后 16 层，rank=8，scale=20，dropout=0。
- 学习率：1e-5；batch size=1；seed=42。
- 最大序列长度：2048；启用梯度检查点；屏蔽 prompt 损失。
- 通过 Validation 按预先记录的规则选择 step450，并在 Test 前锁定。
- 两组评估均关闭 thinking，temperature=0，max_tokens=256。

## 数据与隔离

数据由人工中文模板与参数组合生成。

| 分区 | call | clarify | unsupported | 合计 |
| --- | ---: | ---: | ---: | ---: |
| Train | 360 | 180 | 60 | 600 |
| Validation | 120 | 60 | 20 | 200 |
| Test | 120 | 60 | 20 | 200 |

训练使用 Train 更新参数；Validation 用于比较和选型；
Test 用于固定方案的最终评估。

项目检查了模板、样本及参数的分区隔离。
Test 全部采用明确撤回旧请求、再提出当前请求的单轮表达结构。
精确隔离不能证明广泛的语义独立性或真实业务泛化能力。

## 主要结果

| 分区与指标 | baseline | step450 |
| --- | ---: | ---: |
| Validation：全部自动检查通过 | 176/200（88.0%） | 181/200（90.5%） |
| Validation：合并说明审查后通过 | 172/200（86.0%） | 181/200（90.5%） |
| Test：全部自动检查通过 | 173/200（86.5%） | 137/200（68.5%） |
| Test：合并说明审查后通过 | 159/200（79.5%） | 132/200（66.0%） |

主指标“全部自动检查通过”要求 JSON 对象、接口约束和动作均正确；
参考答案为 call 时，还要求工具名称和全部参数完整匹配。

合并指标进一步要求非调用请求的说明通过人工语义审查。
自动失败样本始终保留在整体分母中。

Validation 的合并通过率提高了 4.5 个百分点，
但当前 Test 的合并通过率下降了 13.5 个百分点。
本轮微调未在当前 Test 上保持 Validation 的提升。

## 错误分析

- 37 条新增退化均表现为信息齐全时的不必要澄清，
  包括重复追问已有的商品编号、订单编号、日期或统计指标。
- 31 条请求两组均失败；其中一个案例是擅自将库存阈值 6.5 改为 6，
  输出满足接口类型要求，但改变了用户请求。
- 10 条合并判定改善来自澄清说明的语义质量。

以上结论来自保存输出的配对分析。
这些现象尚不足以确定具体训练机制或退化原因，
未依据 Test 结果继续选择 checkpoint 或调整参数。

## 环境与复现状态

已使用的本机环境：

- Apple Silicon MacBook Pro M5 Pro，24GB 统一内存。
- Python 3.12.15。
- MLX 0.32.3、MLX-LM 0.32.0。
- Transformers 5.19.0、NumPy 2.5.4。

关键依赖声明在 `pyproject.toml` 中。
原实验环境的可编辑安装与依赖一致性检查已通过。

另在同一台电脑上创建了独立 Conda 环境 `tool_calling_repro`，
按照项目声明重新安装依赖，并通过以下检查：

- `pip check`：未发现依赖冲突。
- 基线开发样例 `train_002`：完成模型加载、生成和评分，全部自动检查通过。
- 正式 step450：恢复 LoRA 结构并加载权重，参数名称与形状检查通过，
  权重 SHA256 与冻结评估方案一致。

上述验证复用了现有项目源码及本地模型、LoRA 权重，
范围限于独立环境安装和基本运行。
全新代码检出、重新训练及整套评估指标的完整复现尚未验证。

基座模型和 LoRA 权重保留在本地，由 Git 忽略规则排除。

## 安装与开发样例运行

适用于 Apple Silicon macOS。以下命令均在项目根目录执行。

### 1. 安装环境与依赖

环境创建与依赖安装会产生下载，预计数十至数百 MB，
具体取决于本地缓存及依赖解析结果。

```bash
conda create -n tool_calling_repro python=3.12.15 pip
conda activate tool_calling_repro
python -m pip install -e .
python -m pip check
```

### 2. 准备本地基座模型

首次准备模型会产生约 4.6 GB 下载。
使用与本轮实验一致的固定 revision：

```bash
hf download mlx-community/Qwen3-8B-4bit \
  --revision 545dc4251c05440727734bcd94334791f6ab0192 \
  --local-dir models/qwen3_8b_4bit
```

已有完整且 revision 一致的本地模型时，可跳过此步骤。

### 3. 运行基线开发样例

```bash
HF_HUB_OFFLINE=1 python -B scripts/run_single_example.py --example-id train_002
```

该命令使用未加载 LoRA 的基线模型，读取开发样例：
“商品 P2001 现在还有多少库存？”

程序输出模型回复、参考答案及自动评分。
预期工具为 `get_inventory`，参数为 `product_id="P2001"`。
这项检查用于验证基本运行流程，不代表整体评估指标复现。

## 工程组织

- `configs/`：工具定义、系统提示词、数据策略和冻结评估方案。
- `data/`：模板、参数、候选样本、选定样本及训练格式数据。
- `src/tool_calling/`：公共数据处理、模型加载和评估逻辑。
- `scripts/`：数据准备、训练、推理、汇总及诊断入口。
- `results/`：原始模型输出与说明审查记录。
- `adapters/`：训练配置、指标记录及本地权重。
- `docs/`：阶段实验记录与最终评估分析。

## 实验记录

- [试验阶段 Validation 记录](docs/pilot_validation_results.md)
- [正式 Validation 记录](docs/formal_validation_results.md)
- [最终 Test 记录与错误分析](docs/formal_test_results.md)

## 限制

本轮仅使用一个训练 seed。
说明审查为导师指导的非盲审，不属于独立盲审或多评审者一致性验证。

尚未开展统计显著性验证和专门耗时评估。
当前结果限于所构建的数据和评估协议，不能直接外推到真实业务系统。

## 许可证

本项目自行编写的代码与文档采用 [MIT 许可证](LICENSE)。
基座模型及第三方依赖遵循各自的许可证。
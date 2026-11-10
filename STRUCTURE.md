# 目录结构约定

本文只描述仓库的**宏观组织约定**：每层目录承担什么职责、文件命名遵循什么规则、层与层之间如何解耦。不涉及任何具体任务实现细节。

---

## 1. 顶层布局

代码统一放在 `src/<包名>/` 下（Python **src-layout**），由 `pyproject.toml` 的 `[tool.uv.build-backend] module-name` 指定包名。仓库根目录只承载配置与工程元信息，不存放可导入的 Python 代码。

```
项目根/
├── src/             # 全部可导入代码（src-layout）
├── scripts/         # 命令行入口脚本
├── tests/           # 测试
├── docs/            # 代码审查、修复归档及课程检查点说明
└── pyproject.toml   # Python 包定义 + entry-point 声明
```

`src-layout` 的意义：杜绝在仓库根目录误 import 到未安装的本地包，确保测试跑的是真实安装路径。

---

## 2. 任务装载通道

任务通过 `pyproject.toml` 的 `[project.entry-points."isaaclab.tasks"]` 注册，由训练、回放、遥操作和评估入口装载。示例 Kit UI 扩展已移除。

包根 `__init__.py` 保持为空，仅写 docstring，避免无图形环境下导入 `omni.*`。真实遥操作的 Kit 输入适配位于 `teleop/kit.py`，在仿真启动后加载。

---

## 3. 包内层级

`src/<包名>/` 按职责分为资产、任务、运行适配及输入设备：

```
src/<包名>/
├── __init__.py                  # 包根：只放 docstring，禁止 import
├── rl/                          # 【运行适配层】训练/回放入口及 Runner 集成
├── teleop/                      # 【输入层】输入状态与设备适配，不加载模型或定义训练奖励
│
├── assets/                      # 【资产层】
│   ├── __init__.py              #   只导出路径常量
│   └── data/                    #   资产文件本体（.gitkeep 占位）
│
└── tasks/                       # 【任务层】
    ├── __init__.py              #   自动扫描并 import 全部子包
    └── <task_name>/             #   一个任务 = 一个目录
        ├── env.py               #     必要的环境生命周期扩展（可选）
        ├── runner.py            #     本任务的独立 Runner 与工厂
        ├── curriculum_state.py  #     任务状态协议，不依赖具体 RL Runner（可选）
        ├── visualization.py     #     与任务语义绑定的附加可视化（可选）
        ├── mdp/                 #     MDP 算子
        └── config/              #     环境配置
            └── <robot>/         #       机器人 / 变体
                ├── __init__.py          # 注册点
                ├── env_cfg.py           # 环境配置
                ├── agents/              # 算法配置
                │   └── <rl_lib>_cfg.py
                └── utils/               # 本体专用资产配置等辅助内容（可选）
```

---

## 4. 核心路径约定

任务目录遵循一条固定的四级路径模板：

```
tasks / <任务名> / config / <机器人名> / agents / <算法库>_cfg.py
```

三层的划分维度各不相同，这是解耦的来源：

| 层级 | 回答的问题 | 变动时机 |
|---|---|---|
| **任务层** `<task_name>` | 学什么技能 | 新增技能时新增目录 |
| **机器人层** `config/<robot>` | 在哪个本体上实现 | 换 embodiment 时新增兄弟目录 |
| **算法层** `agents/<rl_lib>` | 用什么训练 | 换 RL 库时新增一个文件 |

由此得到三条扩展规则：

- 换训练算法 → 只加 `agents/xxx_cfg.py`
- 换机器人本体 → 只加 `config/<新机器人>/`
- 加新任务 → 新增 `tasks/<新任务>/`，并在 `rl/task_registry.py` 绑定该任务自己的 Runner 工厂

任何一条扩展都不需要改动其余层的代码。

---

## 5. 文件命名到角色的映射

结构中每个文件名都承担固定职责，不按个人习惯改名：

| 文件 | 职责约定 |
|---|---|
| `tasks/__init__.py` | 全仓唯一的环境扫描入口，靠递归 import 触发注册 |
| `config/<robot>/__init__.py` | **唯一注册场所**：`gym.register()` 只写在这里 |
| `config/<robot>/env_cfg.py` | 场景 / 动作 / 观测 / 事件 / 奖励 / 终止的汇总配置 |
| `agents/<rl_lib>_cfg.py` | 该 RL 库的 Runner 超参配置 |
| `mdp/<manager类别>.py` | 按 manager 类别切分的自定义算子函数 |
| `assets/__init__.py` | 仅路径常量，禁止放逻辑 |
| `rl/__main__.py`、`rl/runner.py` | 项目运行入口与任务 Runner 分发；不依赖下肢课程字段 |
| `rl/task_registry.py` | 任务族到独立 Runner 工厂的显式绑定与入口能力声明 |
| `tasks/<task_name>/runner.py` | 本任务的 Runner、附加检查点状态和诊断接入；下肢课程只是其中一种实现 |
| `tasks/<task_name>/curriculum_state.py` | 任务状态的导出、校验与导入；不负责模型保存及 CLI |
| `rl/teleop.py` | 遥操作模型加载与回放生命周期，复用项目 Runner |
| `teleop/controls.py`、`teleop/kit.py` | 输入状态与设备订阅；原生输入在仿真启动后创建 |
| `tasks/<task_name>/visualization.py` | 任务专用的几何测量和可视化，避免混入通用输入设备代码 |

两个刻意的设计细节：

1. **注册用字符串路径引用配置类**，而非直接 import。避免注册阶段就加载全部重型配置，同时让 entry-point 可被外部工具静态解析。
2. **`mdp/__init__.py` 采用懒加载 + `.pyi` 桩的组合**：运行时按需导出以降低导入开销，静态检查时由同名 `.pyi` 提供完整符号表与补全。二者必须成对维护。模板任务可先转发通用 MDP 算子，有自定义需求时再增加对应模块。

---

## 6. 扫描机制的边界约定

`tasks/__init__.py` 的递归导入**只处理带 `__init__.py` 的目录**，且接受一个**黑名单参数**。由此产生两条必须遵守的约定：

- 存放算子的目录（如 `mdp/`）**必须列入黑名单**——它不是任务，不含注册语句，被当成任务扫描会产生错误导入。
- 新增的任务目录**不需要手动登记**。只要落在 `tasks/` 下且其 `config/<robot>/__init__.py` 中有注册语句，就会被自动发现。

对应地，测试层的约定是：注册类测试只校验"注册项存在且入口路径字符串正确"，不依赖模拟器启动。

---

## 7. 一句话原则

> 目录名表达**层级职责**，文件名表达**配置归属**，注册集中在 `config/<robot>/__init__.py`，实现细节下沉到 `mdp/` 与 `agents/`。

新增内容时先确定它属于哪一层，再套用第 4 节的路径模板即可，无需修改既有结构。

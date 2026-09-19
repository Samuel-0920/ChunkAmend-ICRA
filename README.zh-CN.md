# ChunkAmend · 真机部署版

**Correcting Action Chunks with Execution Feedback**

[English](README.md) · 简体中文

ChunkAmend 是面向动作块策略的执行时修正方法。它利用已经完成的指令与实际反馈，估计局部六输入、三输出位移模型，为新预测的动作前缀构造有界修正，并在有效的执行边界接纳修正结果。该过程不改变策略模型的权重。

本仓库公开 ChunkAmend 核心算法、仿真与 UF850 真机执行配置、数据采集与训练脚本，以及三个任务的微调 checkpoint。**不公开数据集。**

## 公开内容

| 组件 | 内容 |
|---|---|
| [修正器入口](chunkamend/core.py) | 为仿真或真机数值核心构造选定的 Q6 配置 |
| [仿真实现](chunkamend/simulation/) | 数值核心、异步执行状态与 raw-ready 发布逻辑 |
| [真机实现](chunkamend/hardware/) | 数值核心、预留执行边界的调度器、已完成反馈契约和关节动作坐标转换 |
| [配置文件](configs/) | 选定的修正参数及两种场景的执行设置 |
| [实现说明](docs/IMPLEMENTATION.md) | 修正范围、动作约定和调度语义 |
| [数据采集](real_robot/collection/) | 三任务 Quest/WebXR 遥操作、双相机录制、归位及回合整理脚本 |
| [训练代码](real_robot/training/) | 三任务训练配置、OpenPI 源码快照、预处理及训练入口 |
| [任务配置](real_robot/task_configs/) | 任务描述、原场景参考起点、动作约定及归一化参数 |
| [模型权重](checkpoints/README.md) | 三任务 step-9999 checkpoint、Release 下载及 SHA-256 校验 |
| [环境说明](ENVIRONMENT.md) | 核心依赖版本和各任务训练依赖锁文件 |

两套数值源码分别保留各自场景的实现，由统一入口选择论文配置；策略推理和动作执行通过回调接口接入。

## 仿真与真机

两者均让策略推理与动作执行重叠进行，新动作块的接纳规则有所不同。

| 设置 | 仿真 | UF850 真机 |
|---|---|---|
| 预测长度 | 10 | 15 |
| 标称控制频率 | 20 Hz | 30 Hz |
| 最小查询间隔 | 5 条指令 | 5 条指令 |
| 模型维度 / 实际动作维度 | 32 / 7 | 32 / 7 |
| 接纳规则 | 修正尚未完成时，在下一控制步前发布已就绪的原始动作 | 继续执行旧计划，直到明确预留的切换边界 |
| 修正时机 | 原始动作发布前可接纳仍有效的修正；过期或迟到结果丢弃 | 标称预留 3 个执行步，受剩余预测长度约束；错过边界则采用原始动作 |

选定配置修正前三条平移动作，并在五条动作内完成旋转部分向原始计划的回接。夹爪维度与可写窗口外的动作后缀保持不变。具体参数见 [chunkamend.json](configs/chunkamend.json)、[simulation.json](configs/simulation.json) 和 [hardware.json](configs/hardware.json)。

## 对比方法

| 方法 | 本项目中的设置 | 来源 |
|---|---|---|
| **ChunkAmend** | 基于执行反馈的动作前缀修正，配合各场景的异步接纳调度 | 本项目实现 |
| **RTC** | 完整 32 维引导的异步动作补全 | [RTC 原论文](https://arxiv.org/abs/2506.07339) |
| **TE / stepTE** | 每个逻辑步查询；对同一目标时刻的预测进行时序平均，按从旧到新的顺序使用 `exp(-0.01 i)` 权重 | [ACT](https://arxiv.org/abs/2304.13705) 的时序集成机制 |
| **Vanilla** | 异步执行未经修正的 π0.5 原始预测 | [π0.5 原论文](https://arxiv.org/abs/2504.16054) |

上述名称说明本项目的对比配置。TE 对 π0.5 的预测应用 ACT 式时序集成，不表示将策略模型替换为 ACT。当前公开文件提供上表所述数值与调度组件，完整策略服务和测评运行器不在本包内。

## 发布范围

公开权重对应三个任务：柠檬放入篮子、水果归纳、柠檬放入高脚杯。每个 checkpoint 包含推理参数和匹配的归一化统计，不包含优化器状态。π0.5 基础模型来自 [OpenPI](https://github.com/Physical-Intelligence/openpi)，任务权重由本项目自采示教数据微调得到。

脚本用途和来源见[真机组件说明](real_robot/README.md)。机器人地址、相机绑定及初始位姿是原实验场景的配置，需要按实际设备设置。原始示教、数据集压缩包、录制图片和视频、实验日志均不上传；此次代码发布未在另一台机器上重新执行完整训练或真机验证。

## 许可与归属

项目源码采用 [Apache License 2.0](LICENSE)；checkpoint 同时保留适用的[模型条款与 Gemma 声明](MODEL_LICENSE.md)。本项目实现、上游组件及各方法原论文的归属见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

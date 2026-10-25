# Branch-and-Cut 

使用之前实现的无向模型：COPT 分支定界 + 整数 SEC 懒约束 + 分数解最小割。默认生成 two-step 初始解，再联合优化视点、照片和路径。求解程序及其依赖与原实验包逐字节相同，仅精简批处理入口。

需要 Python 3.9+、numpy、COPT 8.x 及适用的许可证。环境已配好可跳过安装：

```bash
python -m pip install -r requirements.txt
```

本包仅有运行所需程序和本说明，不附带数据、历史结果、对照求解器或测试材料。把自己的输入 JSON 文件放入同级 `data` 文件夹。

## 运行 data 下所有数据

```bash
python -u run_experiment.py \
  --data-dir data --out-dir result \
  --threads 8 --mem-limit-gb 96 \
  --exact --no-time-limit
```

整数 SEC 和分数最小割默认开启。`--exact` 设置最终 joint 的 RelGap、AbsGap 均为 0；`--no-time-limit` 使 two-step 两阶段和 joint 均不限时（COPT 参数 1e20 秒）。不加 `--exact` 时沿用原代码 0.01% 相对 gap。初始化阶段的 gap 保持原设置，仅提供可行解，不固定联合模型决策。

日志实时输出并保存。每个数据完成后自动检查结果约束，以下以 `115.json` 为例：

```text
result/undirected_bc/joint_115.json  求解结果
result/undirected_bc/115.log         实时日志
result/undirected_bc/115.check.json  独立约束检查
result/undirected_bc/115.run.json    参数、输入哈希和运行记录
result/115.comparison.json          运行摘要
```

默认不覆盖已有输出，重跑请更换 `--out-dir`，或明确增加 `--overwrite`。

`runtime` 包括本次 two-step 初始化、joint 建模/求解/核对/清理；不包括输入读取、预处理及最后写盘。`joint_solver_runtime` 仅为 joint 的 COPT 求解时间。结果保留之前约定的全部 22 个基础字段。

## 可选参数与等价性范围

- `--no-two-step-start`：关闭 two-step 初始解。
- `--variants undirected_lazy`：仅整数懒约束，用于与默认分数割版本对比。

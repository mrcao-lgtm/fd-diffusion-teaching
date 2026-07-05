# 一维扩散方程有限差分教学演示

Finite Difference Method for the 1D Diffusion Equation — Teaching Demo

## 概述

本代码用于《大学物理》《物理与工程》等课程中扩散方程的数值教学演示。
对比三种差分格式：

1. **显式前向Euler** — 简单直观，但有稳定性条件 r ≤ 0.5
2. **隐式后向Euler** — 无条件稳定，需解三对角方程组
3. **Crank-Nicolson** — 二阶时间精度 + 无条件稳定

## 用法

```bash
pip install numpy matplotlib
python fd_diffusion.py
```

输出：
- `fig1_solution_comparison.png` — 三种格式的解对比
- `fig2_stability_convergence.png` — 稳定性和收敛性分析
- `fig3_instability_demo.png` — 显式格式 r>0.5 时的数值爆炸演示

## 教学要点

- **r ≤ 0.5 的物理本质**：一个时间步内扩散不能跳过相邻网格点
- **数值爆炸**：违反稳定性条件时产生物理上荒谬的振荡
- **隐式格式的代价**：无条件稳定但需解三对角方程组

## 参考文献

论文: X.Cao. 用有限差分法理解扩散过程[J]. 物理与工程, 2026.

## 许可

MIT License

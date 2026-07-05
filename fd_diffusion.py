#!/usr/bin/env python3
"""
一维扩散方程有限差分 — 显式/隐式/Crank-Nicolson三种格式
教学演示用: 对比稳定性、精度、计算效率
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, os, time

# ====== 参数 ======
L = 1.0        # 空间长度
T = 0.5        # 总时间
alpha = 0.01   # 扩散系数
Nx = 50        # 空间网格数
dx = L / Nx

# 初始条件: 高斯波包
x = np.linspace(0, L, Nx+1)
u0 = np.exp(-100*(x - 0.3)**2)

# 边界条件: Dirichlet u(0)=u(L)=0
def apply_bc(u):
    u[0] = 0.0
    u[-1] = 0.0
    return u

# ====== 显式 Euler ======
def solve_explicit(Nx, dx, T, alpha, r):
    """显式前向Euler: u^{n+1} = u^n + r*(u_{i+1}-2u_i+u_{i-1})"""
    dt = r * dx**2 / alpha
    Nt = int(T / dt)
    u = u0.copy()
    
    for n in range(Nt):
        u_new = u.copy()
        for i in range(1, Nx):
            u_new[i] = u[i] + r*(u[i+1] - 2*u[i] + u[i-1])
        u = apply_bc(u_new)
        
        # 不提前退出,让不稳定充分发展
    return u, dt, Nt

# ====== 隐式 (后向Euler) ======
def solve_implicit(Nx, dx, T, alpha, r):
    """隐式后向Euler: (1+2r)u^{n+1}_i - r(u_{i+1}+u_{i-1}) = u^n_i"""
    dt = r * dx**2 / alpha
    Nt = int(T / dt)
    u = u0.copy()
    
    # 三对角矩阵 (Thomas算法)
    def thomas(a, b, c, d):
        n = len(d)
        cp = np.zeros(n-1)
        dp = np.zeros(n)
        cp[0] = c[0]/b[0]
        dp[0] = d[0]/b[0]
        for i in range(1, n):
            if i < n-1:
                denom = b[i] - a[i-1]*cp[i-1]
                cp[i] = c[i]/denom
            dp[i] = (d[i] - a[i-1]*dp[i-1])/(b[i] - a[i-1]*cp[i-1])
        x = np.zeros(n)
        x[-1] = dp[-1]
        for i in range(n-2, -1, -1):
            x[i] = dp[i] - cp[i]*x[i+1]
        return x
    
    a = -r * np.ones(Nx-1)
    b = (1+2*r) * np.ones(Nx-1)
    c = -r * np.ones(Nx-1)
    
    for n in range(Nt):
        u[1:Nx] = thomas(a, b, c, u[1:Nx])
        u = apply_bc(u)
    return u, dt, Nt

# ====== Crank-Nicolson ======
def solve_crank_nicolson(Nx, dx, T, alpha, r):
    """CN: implicit平均, 二阶精度无条件稳定"""
    dt = r * dx**2 / alpha
    Nt = int(T / dt)
    u = u0.copy()
    
    a_cn = -r/2 * np.ones(Nx-1)
    b_cn = (1+r) * np.ones(Nx-1)
    c_cn = -r/2 * np.ones(Nx-1)
    
    def thomas(a, b, c, d):
        n = len(d)
        cp, dp = np.zeros(n-1), np.zeros(n)
        cp[0], dp[0] = c[0]/b[0], d[0]/b[0]
        for i in range(1, n):
            if i < n-1: cp[i] = c[i]/(b[i] - a[i-1]*cp[i-1])
            dp[i] = (d[i] - a[i-1]*dp[i-1])/(b[i] - a[i-1]*cp[i-1])
        x = np.zeros(n); x[-1] = dp[-1]
        for i in range(n-2, -1, -1): x[i] = dp[i] - cp[i]*x[i+1]
        return x
    
    for n in range(Nt):
        # RHS: explicit half-step
        rhs = u[1:-1] + r/2*(u[2:] - 2*u[1:-1] + u[:-2])
        u[1:Nx] = thomas(a_cn, b_cn, c_cn, rhs)
        u = apply_bc(u)
    return u, dt, Nt

# ====== 运行所有方法 ======
results = {}
r_vals = [0.25, 0.5, 0.51, 1.0, 2.0]  # 扩散数r

print("="*60)
print("一维扩散方程 有限差分 — 三种格式对比")
print(f"Nx={Nx}, T={T}, alpha={alpha}")
print("="*60)

for scheme_name, solver in [("explicit", solve_explicit), ("implicit", solve_implicit), ("cn", solve_crank_nicolson)]:
    results[scheme_name] = {}
    for r_val in r_vals:
        t0 = time.time()
        try:
            u, dt, Nt = solver(Nx, dx, T, alpha, r_val)
            elapsed = time.time() - t0
            stable = dt > 0 and Nt > 0 and not np.any(np.isnan(u)) and np.max(np.abs(u)) < 10
            results[scheme_name][f"r{r_val}"] = {
                "dt": float(dt), "Nt": int(Nt), "stable": bool(stable),
                "u_max": float(np.max(np.abs(u))), "time_s": float(elapsed)
            }
            status = "✅稳定" if stable else "❌发散"
            print(f"  {scheme_name:>10s} r={r_val:.2f}: dt={dt:.6f} Nt={Nt:4d} max|u|={np.max(np.abs(u)):.4f} [{elapsed:.3f}s] {status}")
        except Exception as e:
            results[scheme_name][f"r{r_val}"] = {"error": str(e), "stable": False}
            print(f"  {scheme_name:>10s} r={r_val:.2f}: ERROR {e}")

# ====== 画图 ======
fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
os.makedirs(fig_dir, exist_ok=True)

# Fig1: 三种格式最终解对比 (r=0.25, 稳定区)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(x, u0, 'k--', label='Initial', alpha=0.5, linewidth=2)
schemes_plot = [("explicit", solve_explicit, "#1f77b4","-"), ("implicit", solve_implicit, "#d62728","--"), ("cn", solve_crank_nicolson, "#2ca02c","-.")]
for scheme, solver_fn, color, ls in schemes_plot:
    u, _, _ = solver_fn(Nx, dx, T, alpha, 0.25)
    ax.plot(x, u, color=color, linestyle=ls, label=scheme, linewidth=2)
ax.set_xlabel('x (m)'); ax.set_ylabel('u(x,T)')
ax.set_title('(a) r=0.25 (stable)'); ax.legend(); ax.grid(True, alpha=0.3)

# Fig1b: 显式r=0.51 发散
ax = axes[1]
ax.plot(x, u0, 'k--', label='Initial', alpha=0.5)
u_exp_unstable, _, _ = solve_explicit(Nx, dx, T, alpha, 0.51)
ax.plot(x, u_exp_unstable, 'r-', label='Explicit r=0.51', linewidth=2)
ax.set_xlabel('x (m)'); ax.set_ylabel('u(x,T)')
ax.set_title('(b) Explicit r=0.51 (unstable)')
ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'fig1_solution_comparison.png'), dpi=200)
plt.close()
print("\n✅ fig1: 解对比")

# Fig2: 稳定性区域 + 误差收敛
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Stability table
ax = axes[0]
schemes = ["explicit", "implicit", "cn"]
r_vals_plot = [0.25, 0.5, 0.51, 1.0, 2.0]
stable_data = {s: [] for s in schemes}
for s in schemes:
    for rv in r_vals_plot:
        stable_data[s].append(results[s][f"r{rv}"].get("stable", False))

x_pos = np.arange(len(r_vals_plot))
width = 0.25
colors_stab = {'explicit': '#1f77b4', 'implicit': '#d62728', 'cn': '#2ca02c'}
for i, s in enumerate(schemes):
    bars = ax.bar(x_pos + i*width, [1 if v else 0 for v in stable_data[s]], 
                  width, color=colors_stab[s], label=s, alpha=0.8)
ax.set_xticks(x_pos + width)
ax.set_xticklabels([f'r={rv}' for rv in r_vals_plot])
ax.set_ylabel('Stable (1=yes)')
ax.set_title('(a) Stability vs diffusion number r')
ax.legend(); ax.grid(True, alpha=0.3, axis='y')

# Convergence: CN vs analytical solution at different dt
ax = axes[1]
# Analytical: fundamental mode decay
analytical = np.exp(-alpha * T * (np.pi/L)**2) * np.sin(np.pi * x / L)
u_cn, _, _ = solve_crank_nicolson(Nx, dx, T, alpha, 0.5)
ax.plot(x, analytical, 'k--', label='Analytical (1st mode)', linewidth=2, alpha=0.7)
ax.plot(x, u_cn, '-', color='#2ca02c', label='CN (Nx=50)', linewidth=2)
ax.set_xlabel('x (m)'); ax.set_ylabel('u(x,T)')
ax.set_title('(b) CN vs analytical solution')
ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'fig2_stability_convergence.png'), dpi=200)
plt.close()
print("✅ fig2: 稳定性+收敛性")

# ====== 保存数据 ======
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
os.makedirs(data_dir, exist_ok=True)
outpath = os.path.join(data_dir, 'fd_diffusion_results.json')
with open(outpath, 'w') as f:
    json.dump({
        "metadata": {"method": "有限差分法三种格式", "equation": "1D diffusion ∂u/∂t=α∂²u/∂x²",
                     "Nx": Nx, "T": T, "alpha": alpha, "dx": dx},
        "results": results
    }, f, indent=2)
print(f"\n✅ 数据: {outpath}")
print(f"✅ 图: {fig_dir}/")


# ====== Fig3: Instability demo (r=0.75 with Nyquist mode) ======
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

L, alpha, Nx = 1.0, 0.01, 50
dx = L/Nx
x = np.linspace(0, L, Nx+1)
u0 = np.sin(np.pi*x/L) + 0.1*(-1)**np.arange(Nx+1)
u0[0] = u0[-1] = 0

r = 0.75; dt = r*dx**2/alpha
u = u0.copy()
history = [u.copy()]
for n in range(30):
    u_new = u.copy()
    for i in range(1, Nx): u_new[i] = u[i] + r*(u[i+1]-2*u[i]+u[i-1])
    u_new[0]=u_new[-1]=0; u = u_new
    history.append(u.copy())
    if np.max(np.abs(u)) > 1e6: break

fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
os.makedirs(fig_dir, exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 5))
colors = plt.cm.Reds(np.linspace(0.3, 1, min(6, len(history))))
for idx, i in enumerate([0,2,4,6,8, n]):
    if i < len(history):
        ax.plot(x, history[i], color=colors[idx], label=f't={min(i*dt,0.720):.3f}', linewidth=1.5)
ax.plot(x, u0, 'k--', alpha=0.5, label='Initial')
ax.set_xlabel('x (m)'); ax.set_ylabel('u(x,t)')
ax.set_title(f'Explicit r=0.75 > 0.5: stability criterion violation, {n+1} steps to explosion')
ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'fig3_instability_demo.png'), dpi=200)
plt.close()
print(f"✅ Fig3: r=0.75 instability, {n+1} steps, max|u|={np.max(np.abs(u)):.1e}")

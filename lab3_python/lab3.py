"""Лабораторная работа 3. Динамические аналитические модели.

Вариант 12: общие задания 1-3 и индивидуальное задание 4 (крен самолета).
Решатели:
  ode23  - solve_ivp(method="RK23"), явный метод Богацки-Шампина 3(2);
  ode113 - ode("vode", method="adams"), многошаговый метод Адамса
           переменного порядка (предиктор-корректор).
Все системы линейные с постоянными коэффициентами, поэтому программа
дополнительно строит точное решение через матричную экспоненту и по нему
оценивает погрешность каждого решателя.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import sys
import time
import warnings

import numpy as np
from scipy.integrate import ode, solve_ivp
from scipy.interpolate import CubicHermiteSpline
from scipy.linalg import expm

RESULT_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_TOL = (1e-3, 1e-6)      # RelTol и AbsTol по умолчанию
STRICT_TOL = (1e-8, 1e-10)
TABLE_POINTS = 6
SETTLE_EPS = 0.01               # граница установившегося режима, град
SOLVERS = ("ode23", "ode113")


@dataclass
class Task:
    key: str
    title: str
    equations: list
    names: list
    labels: list
    A: np.ndarray
    G: np.ndarray        # x' = A x + G e(t)
    S: np.ndarray        # e' = S e - источник внешнего воздействия
    e0: np.ndarray
    x0: tuple
    span: tuple
    alt_x0: tuple
    alt_span: tuple
    unit: str = ""

    @property
    def n(self):
        return len(self.names)

    def forcing(self, t):
        return self.G @ (expm(self.S * t) @ self.e0) if self.S.size > 1 else self.G @ self.e0

    def rhs(self, t, x):
        return self.A @ x + self.forcing(t)

    def augmented(self):
        m = self.S.shape[0]
        top = np.hstack([self.A, self.G])
        bottom = np.hstack([np.zeros((m, self.n)), self.S])
        return np.vstack([top, bottom])

    def exact(self, x0, times):
        M = self.augmented()
        z0 = np.concatenate([np.asarray(x0, float), self.e0])
        return np.array([(expm(M * t) @ z0)[: self.n] for t in np.atleast_1d(times)])


def harmonic_forcing():
    return np.array([[0.0, -1.0], [1.0, 0.0]]), np.array([1.0, 0.0])


def build_tasks():
    S_cos, e_cos = harmonic_forcing()
    const_S, const_e = np.zeros((1, 1)), np.array([1.0])
    roll_A = np.array([[-10.0, -66.97, -10.74],
                       [0.0, 0.0, 1.0],
                       [5.0, 0.0, -1.0]])
    return [
        Task("task1", "Задание 1. Система линейных ОДУ",
             ["dx1/dt = -3·x1 + 10,  x1(0) = 0",
              "dx2/dt = x1 - 2·x2,   x2(0) = 0",
              "dx3/dt = 4·x2 - x3,   x3(0) = 0"],
             ["x1", "x2", "x3"], ["$x_1$", "$x_2$", "$x_3$"],
             np.array([[-3.0, 0, 0], [1, -2, 0], [0, 4, -1]]),
             np.array([[10.0], [0], [0]]), const_S, const_e,
             (0, 0, 0), (0, 10), (5, -2, 10), (0, 3)),
        Task("task2", "Задание 2. Уравнение второго порядка",
             ["x'' + 4·x' + 2·x = cos(t),  x(0) = 0,  x'(0) = 0",
              "замена x1 = x, x2 = x':",
              "dx1/dt = x2",
              "dx2/dt = -2·x1 - 4·x2 + cos(t)"],
             ["x1", "x2"], ["$x_1=x$", "$x_2=\\dot{x}$"],
             np.array([[0.0, 1], [-2, -4]]),
             np.array([[0.0, 0], [1, 0]]), S_cos, e_cos,
             (0, 0), (0, 20), (1, -1), (0, 40)),
        Task("task3", "Задание 3. Матричная система dx/dt = A·x + B·u",
             ["A = [[-1, 0, 0], [0, 3, -5], [0, 4, -5]]",
              "B = [[12, 0, 0], [0, 0, 0], [0, 0, 0]],  u = (1, 1, 1)ᵀ",
              "x(0) = (0, 0, 0)ᵀ"],
             ["x1", "x2", "x3"], ["$x_1$", "$x_2$", "$x_3$"],
             np.array([[-1.0, 0, 0], [0, 3, -5], [0, 4, -5]]),
             np.array([[12.0, 0, 0], [0, 0, 0], [0, 0, 0]]) @ np.ones((3, 1)),
             const_S, const_e,
             (0, 0, 0), (0, 9), (2, 1, 1), (0, 15)),
        Task("task4", "Задание 4 (вариант 12). Возмущенное движение самолета по крену",
             ["dδэ/dt = -u,                 δэ(0) = 15 град",
              "dγ/dt = ωx1,                 γ(0) = 2 град",
              "dωx1/dt = -ωx1/Tγ + Nэ·δэ/Tγ,  ωx1(0) = 7 град/с",
              "u = iδ·δэ + iω·ωx1 + iγ·γ,  iδ = 10, iω = 10,74, iγ = 66,97",
              "Tγ = 1 с, Nэ = 5 1/с"],
             ["delta", "gamma", "omega"], ["$\\delta_э$, град", "$\\gamma$, град",
                                            "$\\omega_{x1}$, град/с"],
             roll_A, np.zeros((3, 1)), const_S, const_e,
             (15, 2, 7), (0, 8), (-10, 5, -3), (0, 3), unit="град"),
    ]


ROLL_GAINS = np.array([10.0, 66.97, 10.74])       # порядок состояния: δ, γ, ω
ROLL_OPEN_A = np.array([[0.0, 0, 0], [0, 0, 1], [5, 0, -1]])


@dataclass
class Solution:
    solver: str
    tol: tuple
    t: np.ndarray
    x: np.ndarray
    nfev: int
    seconds: float
    dense: object        # интерполянт по узлам этого же прогона

    def at(self, points):
        return np.asarray(self.dense(np.asarray(points, float)))


class Counter:
    def __init__(self, f):
        self.f, self.calls = f, 0

    def __call__(self, t, x):
        self.calls += 1
        return self.f(t, x)


def solve_ode23(task, x0, span, tol):
    f = Counter(task.rhs)
    start = time.perf_counter()
    sol = solve_ivp(f, span, np.asarray(x0, float), method="RK23",
                    rtol=tol[0], atol=tol[1], dense_output=True)
    if not sol.success:
        raise RuntimeError(sol.message)
    return Solution("ode23", tol, sol.t, sol.y.T, f.calls, time.perf_counter() - start,
                    lambda t: sol.sol(t).T)


def solve_ode113(task, x0, span, tol):
    """Шаги метода Адамса в режиме step=True; значения между узлами и в конце
    интервала - кубическая эрмитова интерполяция по узлам и производным."""
    f = Counter(task.rhs)
    start = time.perf_counter()
    r = ode(f).set_integrator("vode", method="adams", rtol=tol[0], atol=tol[1],
                              nsteps=100000, order=12)
    r.set_initial_value(np.asarray(x0, float), span[0])
    ts, xs = [span[0]], [np.asarray(x0, float)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        while ts[-1] < span[1]:
            r.integrate(span[1], step=True)
            if not r.successful():
                raise RuntimeError("Адамс (vode): шаг не выполнен")
            ts.append(r.t)
            xs.append(r.y.copy())
    seconds = time.perf_counter() - start
    ts, xs = np.array(ts), np.array(xs)
    dxs = np.array([task.rhs(t, x) for t, x in zip(ts, xs)])
    spline = CubicHermiteSpline(ts, xs, dxs, axis=0)
    inside = ts < span[1]
    t_out = np.append(ts[inside], span[1])
    x_out = np.vstack([xs[inside], spline(span[1])])
    return Solution("ode113", tol, t_out, x_out, f.calls, seconds, spline)


SOLVE = {"ode23": solve_ode23, "ode113": solve_ode113}


def stats(task, sol, x0):
    err = np.abs(sol.x - task.exact(x0, sol.t))
    steps = np.diff(sol.t)
    return {
        "solver": sol.solver, "rtol": sol.tol[0], "atol": sol.tol[1],
        "steps": int(len(sol.t) - 1), "nfev": int(sol.nfev),
        "h_min": float(steps.min()), "h_max": float(steps.max()),
        "max_err": [float(v) for v in err.max(axis=0)],
        "final": [float(v) for v in sol.x[-1]],
        "ms": round(sol.seconds * 1000, 2),
    }


def steady_state(task):
    if task.key == "task2":
        a, b = 1 / 17, 4 / 17
        return {"kind": "periodic", "x": f"{a:.6f}·cos t + {b:.6f}·sin t",
                "amplitude": float(np.hypot(a, b))}
    b = task.G @ task.e0
    return {"kind": "constant", "x": [float(v) for v in np.linalg.solve(task.A, -b)]}


def settle_time(t, x, eps=SETTLE_EPS):
    outside = np.where(np.abs(x).max(axis=1) > eps)[0]
    if len(outside) == 0:
        return 0.0
    if outside[-1] == len(t) - 1:
        return None
    return float(t[outside[-1] + 1])


# ---------- вывод ----------

class Tee:
    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8")

    def __call__(self, text=""):
        print(text)
        self.file.write(text + "\n")


def ascii_table(headers, rows):
    rows = [[str(v) for v in row] for row in rows]
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    line = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def fmt(row):
        return "| " + " | ".join(v.rjust(w) for v, w in zip(row, widths)) + " |"
    return "\n".join([line, fmt(headers), line, *map(fmt, rows), line])


def num(v, digits=6):
    return f"{v:.{digits}f}"


def sci(v):
    return f"{v:.2e}"


# ---------- графики ----------

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "font.size": 10,
                         "axes.grid": True, "grid.alpha": 0.35,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "legend.fontsize": 8, "legend.framealpha": 0.9})
    return plt


class Figures:
    def __init__(self, plt):
        self.plt, self.items, self.number = plt, [], 0

    def save(self, fig, name, caption):
        self.number += 1
        file = f"{self.number:02d}_{name}.png"
        fig.tight_layout()
        fig.savefig(RESULT_DIR / file, bbox_inches="tight")
        self.plt.close(fig)
        self.items.append({"file": file, "caption": caption})
        return file


def plot_time(figs, task, sols, title, name, caption):
    plt = figs.plt
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, label in enumerate(task.labels):
        a, b = sols["ode23"], sols["ode113"]
        ax.plot(a.t, a.x[:, i], "-o", ms=2.5, lw=1.4, color=COLORS[i], label=f"{label} — ode23")
        ax.plot(b.t, b.x[:, i], "--", lw=1.2, color=COLORS[i], alpha=0.75,
                label=f"{label} — ode113")
    ax.set_xlabel("t, с" if task.unit else "t")
    ax.set_ylabel("значения координат")
    ax.set_title(title)
    ax.legend(ncol=2)
    return figs.save(fig, name, caption)


def plot_variants(figs, task, variants, name, caption):
    plt = figs.plt
    fig, axes = plt.subplots(task.n, 1, figsize=(8, 2.1 * task.n + 0.6), sharex=True)
    styles = ["-", "--", ":"]
    for i, ax in enumerate(axes):
        for k, (label, sol) in enumerate(variants):
            ax.plot(sol.t, sol.x[:, i], styles[k % 3], lw=1.5, color=COLORS[k], label=label)
        ax.set_ylabel(task.labels[i])
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("t, с" if task.unit else "t")
    fig.suptitle(f"{task.title.split('.')[0]}: изменение интервала и начальных условий",
                 fontsize=11)
    return figs.save(fig, name, caption)


def smooth(sol, count=800):
    return sol.at(np.linspace(sol.t[0], sol.t[-1], count))


def plot_phase3d(figs, task, variants, name, caption):
    plt = figs.plt
    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(projection="3d")
    for k, (label, sol) in enumerate(variants):
        xs = smooth(sol)
        ax.plot(xs[:, 0], xs[:, 1], xs[:, 2], lw=1.6, color=COLORS[k], label=label)
        ax.scatter(*sol.x[0], color=COLORS[k], s=25, marker="o")
        ax.scatter(*sol.x[-1], color=COLORS[k], s=40, marker="*")
    ss = steady_state(task)
    if ss["kind"] == "constant":
        ax.scatter(*ss["x"], color="black", s=60, marker="X", label="установившееся состояние")
    ax.set_xlabel(task.labels[0])
    ax.set_ylabel(task.labels[1])
    ax.set_zlabel(task.labels[2])
    ax.set_title("Фазовая траектория в пространстве координат")
    ax.legend(loc="upper left")
    return figs.save(fig, name, caption)


def plot_phase2d(figs, task, i, j, variants, name, caption, extra=None):
    plt = figs.plt
    fig, ax = plt.subplots(figsize=(6.4, 5))
    for k, (label, sol) in enumerate(variants):
        xs = smooth(sol)
        ax.plot(xs[:, i], xs[:, j], lw=1.5, color=COLORS[k], label=label)
        ax.plot(*xs[0, [i, j]], "o", color=COLORS[k])
        ax.annotate("", xy=xs[12, [i, j]], xytext=xs[8, [i, j]],
                    arrowprops=dict(arrowstyle="->", color=COLORS[k]))
    if extra:
        extra(ax)
    ax.axhline(0, color="gray", lw=0.7)
    ax.axvline(0, color="gray", lw=0.7)
    ax.set_xlabel(task.labels[i])
    ax.set_ylabel(task.labels[j])
    ax.set_title("Фазовая плоскость")
    ax.legend()
    return figs.save(fig, name, caption)


def plot_errors(figs, task, runs, x0, name, caption):
    plt = figs.plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for sol in runs:
        err = np.abs(sol.x - task.exact(x0, sol.t)).max(axis=1)
        label = f"{sol.solver}, RelTol={sol.tol[0]:.0e}"
        style = "-" if sol.tol == DEFAULT_TOL else "--"
        color = COLORS[0] if sol.solver == "ode23" else COLORS[1]
        ax1.semilogy(sol.t[1:], np.maximum(err[1:], 1e-17), style, lw=1.3, color=color,
                     label=label)
        ax2.semilogy(sol.t[1:], np.diff(sol.t), style, marker=".", ms=3, lw=1, color=color,
                     label=label)
    ax1.set_ylabel("max |x - x_точн|")
    ax1.set_title("Погрешность относительно точного решения (матричная экспонента)")
    ax2.set_ylabel("шаг h")
    ax2.set_xlabel("t, с" if task.unit else "t")
    ax2.set_title("Величина шага интегрирования")
    ax1.legend(ncol=2)
    return figs.save(fig, name, caption)


def plot_roll(figs, task, sol, settle, name, caption):
    plt = figs.plt
    keep = sol.t <= 10
    t, x = sol.t[keep], sol.x[keep]
    u = x @ ROLL_GAINS
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    series = [(x[:, 1], task.labels[1], "Угол крена"),
              (x[:, 2], task.labels[2], "Угловая скорость крена"),
              (x[:, 0], task.labels[0], "Отклонение элеронов"),
              (u, "u, град/с", "Сигнал управления приводом")]
    for ax, (y, label, title), color in zip(axes.flat, series, COLORS):
        ax.plot(t, y, lw=1.6, color=color)
        ax.fill_between(t, -SETTLE_EPS, SETTLE_EPS, color="gray", alpha=0.25)
        if settle is not None:
            ax.axvline(settle, color="black", ls=":", lw=1)
        ax.set_title(title)
        ax.set_ylabel(label)
    for ax in axes[1]:
        ax.set_xlabel("t, с")
    if settle is not None:
        axes[0, 0].annotate(f"T уст ≈ {settle:.2f} с", xy=(settle, 0),
                            xytext=(settle + 0.4, 0.9 * x[:, 1].max()),
                            arrowprops=dict(arrowstyle="->"))
    fig.suptitle("Замкнутая система стабилизации крена (ode113)", fontsize=11)
    return figs.save(fig, name, caption)


def plot_open_loop(figs, task, closed, open_t, open_x, name, caption):
    plt = figs.plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))
    keep = closed.t <= open_t[-1]
    closed = Solution(closed.solver, closed.tol, closed.t[keep], closed.x[keep], 0, 0, None)
    ax1.plot(closed.t, closed.x[:, 1], lw=1.6, color=COLORS[0], label="с регулятором")
    ax1.plot(open_t, open_x[:, 1], "--", lw=1.6, color=COLORS[1], label="без регулятора (u = 0)")
    ax1.set_title("Угол крена γ(t)")
    ax1.set_xlabel("t, с")
    ax1.set_ylabel("γ, град")
    ax1.legend()
    ax2.plot(closed.t, closed.x[:, 2], lw=1.6, color=COLORS[0], label="с регулятором")
    ax2.plot(open_t, open_x[:, 2], "--", lw=1.6, color=COLORS[1], label="без регулятора (u = 0)")
    ax2.set_title("Угловая скорость ωx1(t)")
    ax2.set_xlabel("t, с")
    ax2.set_ylabel("ωx1, град/с")
    ax2.legend()
    return figs.save(fig, name, caption)


# ---------- расчет ----------

def check_model(task):
    """Правая часть системы должна совпадать с точным решением через экспоненту."""
    t = np.linspace(task.span[0], task.span[1], 7)
    x = task.exact(task.x0, t)
    h = 1e-6
    deriv = (task.exact(task.x0, t + h) - task.exact(task.x0, t - h)) / (2 * h)
    rhs = np.array([task.rhs(ti, xi) for ti, xi in zip(t, x)])
    return float(np.abs(deriv - rhs).max())


def run_task(task, out, figs):
    out("=" * 78)
    out(task.title)
    out("=" * 78)
    for line in task.equations:
        out("  " + line)
    eig = np.linalg.eigvals(task.A)
    out("\nМатрица системы A:")
    for row in task.A:
        out("  [" + "  ".join(f"{v:8.3f}" for v in row) + " ]")
    out("Собственные значения A: " + ",  ".join(
        f"{z.real:.4f}{z.imag:+.4f}i" if abs(z.imag) > 1e-12 else f"{z.real:.4f}" for z in eig))
    stable = bool(np.all(eig.real < 0))
    out("Все Re(λ) < 0 -> система асимптотически устойчива" if stable
        else "Есть Re(λ) >= 0 -> система неустойчива")
    ss = steady_state(task)
    if ss["kind"] == "constant":
        out("Установившееся состояние x* = -A⁻¹·b: (" + ", ".join(num(v, 4) for v in ss["x"]) + ")")
    else:
        out(f"Установившийся вынужденный режим: x(t) = {ss['x']}, "
            f"амплитуда {ss['amplitude']:.6f}")
    out(f"Самопроверка модели (производная точного решения - правая часть): "
        f"{check_model(task):.1e}")

    plan = [("базовый", task.x0, task.span),
            ("другой интервал", task.x0, task.alt_span),
            ("другие нач. условия", task.alt_x0, task.span)]
    runs, stat_rows = {}, []
    for label, x0, span in plan:
        for solver in SOLVERS:
            sol = SOLVE[solver](task, x0, span, DEFAULT_TOL)
            runs[(label, solver)] = sol
            st = stats(task, sol, x0)
            st.update(variant=label, x0=list(x0), span=list(span))
            stat_rows.append(st)
    for solver in SOLVERS:
        sol = SOLVE[solver](task, task.x0, task.span, STRICT_TOL)
        runs[("строгая точность", solver)] = sol
        st = stats(task, sol, task.x0)
        st.update(variant="строгая точность", x0=list(task.x0), span=list(task.span))
        stat_rows.append(st)

    points = np.linspace(*task.span, TABLE_POINTS)
    values = {s: runs[("базовый", s)].at(points) for s in SOLVERS}
    exact = task.exact(task.x0, points)
    compare = []
    for k, t in enumerate(points):
        for i, name in enumerate(task.names):
            a, b, e = values["ode23"][k, i], values["ode113"][k, i], exact[k, i]
            compare.append({"t": float(t), "coord": name, "ode23": float(a), "ode113": float(b),
                            "exact": float(e), "diff": float(abs(a - b)),
                            "err23": float(abs(a - e)), "err113": float(abs(b - e))})

    out(f"\nПокоординатное сравнение (базовый вариант, RelTol={DEFAULT_TOL[0]:.0e}, "
        f"AbsTol={DEFAULT_TOL[1]:.0e}):")
    out(ascii_table(["t", "коорд.", "ode23", "ode113", "точное", "|ode23-ode113|"],
                    [[num(r["t"], 2), r["coord"], num(r["ode23"]), num(r["ode113"]),
                      num(r["exact"]), sci(r["diff"])] for r in compare]))

    out("\nСтатистика решателей:")
    out(ascii_table(["вариант", "решатель", "RelTol", "шагов", "вызовов f",
                     "h min", "h max", "max погрешность", "мс"],
                    [[r["variant"], r["solver"], f"{r['rtol']:.0e}", r["steps"], r["nfev"],
                      sci(r["h_min"]), sci(r["h_max"]), sci(max(r["max_err"])), r["ms"]]
                     for r in stat_rows]))

    out("\nЗначения в конце интервала:")
    rows = []
    for label, x0, span in plan:
        ex = task.exact(x0, [span[1]])[0]
        for i, name in enumerate(task.names):
            rows.append([label, f"[{span[0]}; {span[1]}]", name,
                         num(runs[(label, 'ode23')].x[-1, i]),
                         num(runs[(label, 'ode113')].x[-1, i]), num(ex[i])])
    out(ascii_table(["вариант", "интервал", "коорд.", "ode23", "ode113", "точное"], rows))
    final_rows = [dict(zip(["variant", "span", "coord", "ode23", "ode113", "exact"], r))
                  for r in rows]

    base = {s: runs[("базовый", s)] for s in SOLVERS}
    short = task.title.split(".")[0]
    variants = [(f"базовый: x0={task.x0}, t∈{list(task.span)}", base["ode113"]),
                (f"интервал t∈{list(task.alt_span)}", runs[("другой интервал", "ode113")]),
                (f"x0={task.alt_x0}", runs[("другие нач. условия", "ode113")])]
    phase_variants = [variants[0], variants[2]]
    extra = {}
    figures = []
    figures.append(plot_time(figs, task, base, f"{short}: решение решателями ode23 и ode113",
                             f"{task.key}_time",
                             f"{short} — решение при базовых условиях (ode23 и ode113)"))
    figures.append(plot_variants(figs, task, variants, f"{task.key}_variants",
                                 f"{short} — решения при изменении интервала и начальных условий"))
    if task.key in ("task1", "task3"):
        figures.append(plot_phase3d(figs, task, phase_variants, f"{task.key}_phase3d",
                                    f"{short} — фазовая траектория в пространстве (x1, x2, x3)"))
    if task.key == "task2":
        a, b = 1 / 17, 4 / 17
        tt = np.linspace(0, 2 * np.pi, 400)

        def cycle(ax):
            ax.plot(a * np.cos(tt) + b * np.sin(tt), -a * np.sin(tt) + b * np.cos(tt),
                    color="black", lw=1, ls=":", label="предельный цикл (вынужд. режим)")
        figures.append(plot_phase2d(figs, task, 0, 1, phase_variants, f"{task.key}_phase",
                                    f"{short} — фазовая плоскость (x, x')", cycle))
    if task.key == "task3":
        figures.append(plot_phase2d(figs, task, 1, 2, phase_variants, f"{task.key}_phase23",
                                    f"{short} — фазовая плоскость (x2, x3): затухающая спираль"))
    if task.key == "task4":
        long_sol = solve_ode113(task, task.x0, (0, 20), STRICT_TOL)
        settle = settle_time(long_sol.t, long_sol.x)
        alt_long = solve_ode113(task, task.alt_x0, (0, 20), STRICT_TOL)
        settle_alt = settle_time(alt_long.t, alt_long.x)
        u = long_sol.x @ ROLL_GAINS
        peak = np.abs(long_sol.x).max(axis=0)
        out(f"\nВремя установления (все |координаты| <= {SETTLE_EPS} град): "
            f"T = {settle:.3f} с; при других нач. условиях T = {settle_alt:.3f} с")
        out(f"Максимумы модулей: |δэ| = {peak[0]:.3f} град, |γ| = {peak[1]:.3f} град, "
            f"|ωx1| = {peak[2]:.3f} град/с, |u| = {np.abs(u).max():.3f} град/с")
        gamma = long_sol.x[:, 1]
        overshoot = float(-gamma.min()) if gamma.min() < 0 else 0.0
        out(f"Максимальный угол крена {gamma.max():.3f} град достигается при "
            f"t = {long_sol.t[gamma.argmax()]:.3f} с; заброс в обратную сторону {overshoot:.3f} град")
        zeta = -eig.real / np.abs(eig)
        out("Коэффициенты затухания корней: " + ", ".join(f"{z:.3f}" for z in zeta))
        open_task = Task(**{**task.__dict__, "A": ROLL_OPEN_A})
        open_t = np.linspace(0, 8, 400)
        open_x = open_task.exact(task.x0, open_t)
        out(f"Без регулятора (u = 0) при t = 8 с: γ = {open_x[-1, 1]:.1f} град, "
            f"ωx1 = {open_x[-1, 2]:.2f} град/с -> крен неограниченно растет")
        extra = {"settle": settle, "settle_alt": settle_alt, "peak": peak.tolist(),
                 "u_max": float(np.abs(u).max()), "gamma_max": float(gamma.max()),
                 "gamma_max_t": float(long_sol.t[gamma.argmax()]), "overshoot": overshoot,
                 "zeta": zeta.tolist(), "open_gamma8": float(open_x[-1, 1]),
                 "open_omega8": float(open_x[-1, 2])}
        figures.append(plot_roll(figs, task, long_sol, settle, "task4_roll",
                                 "Задание 4 — переходные процессы δэ, γ, ωx1 и сигнал u"))
        figures.append(plot_phase2d(figs, task, 1, 2, phase_variants, "task4_phase",
                                    "Задание 4 — фазовая плоскость (γ, ωx1)"))
        figures.append(plot_open_loop(figs, task, long_sol, open_t, open_x, "task4_open",
                                      "Задание 4 — сравнение движения с регулятором и без него"))
    figures.append(plot_errors(figs, task,
                               [base["ode23"], base["ode113"], runs[("строгая точность", "ode23")],
                                runs[("строгая точность", "ode113")]], task.x0,
                               f"{task.key}_errors",
                               f"{short} — погрешность решателей и величина шага"))
    out("\nГрафики: " + ", ".join(figures))
    out()
    return {
        "key": task.key, "title": task.title, "equations": task.equations,
        "names": task.names, "A": task.A.tolist(),
        "eigenvalues": [[float(z.real), float(z.imag)] for z in eig], "stable": stable,
        "steady": ss, "x0": list(task.x0), "span": list(task.span),
        "alt_x0": list(task.alt_x0), "alt_span": list(task.alt_span),
        "compare": compare, "stats": stat_rows, "final": final_rows,
        "figures": [f for f in figs.items if f["file"] in figures], "extra": extra,
    }


def main():
    RESULT_DIR.mkdir(exist_ok=True)
    for old in RESULT_DIR.glob("*.png"):
        old.unlink()
    out = Tee(RESULT_DIR / "results.txt")
    plt = setup_matplotlib()
    figs = Figures(plt)
    out("Лабораторная работа 3. Динамические аналитические модели. Вариант 12")
    out(f"Python {sys.version.split()[0]}, numpy {np.__version__}")
    out("ode23  = RK23 (Богацки-Шампин 3(2)), ode113 = Адамс переменного порядка (VODE)")
    out(f"Точность по умолчанию: RelTol={DEFAULT_TOL[0]:.0e}, AbsTol={DEFAULT_TOL[1]:.0e}; "
        f"строгая: RelTol={STRICT_TOL[0]:.0e}, AbsTol={STRICT_TOL[1]:.0e}\n")
    results = [run_task(task, out, figs) for task in build_tasks()]
    summary = [[r["key"], s["variant"], s["solver"], s["steps"], sci(max(s["max_err"]))]
               for r in results for s in r["stats"] if s["variant"] in ("базовый", "строгая точность")]
    out("ИТОГ: сравнение решателей по всем заданиям")
    out(ascii_table(["задание", "вариант", "решатель", "шагов", "max погрешность"], summary))
    data = {"default_tol": DEFAULT_TOL, "strict_tol": STRICT_TOL, "tasks": results}
    (RESULT_DIR / "results.json").write_text(json.dumps(data, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
    out(f"\nРезультаты сохранены в папке {RESULT_DIR}")
    out.file.close()


if __name__ == "__main__":
    main()

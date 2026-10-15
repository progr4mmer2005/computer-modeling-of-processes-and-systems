"""Лабораторная работа 4. Построение аналитической модели по результатам эксперимента.

Вариант 12, задание 12: Sr_волосы = f(Sr_воды), прогноз для Sr_воды = 0,424.
Программа повторяет ручной расчет из методички (суммы, коэффициенты прямой и
обратной регрессии, r, R², дисперсии, доверительные интервалы), строит все пять
моделей линии тренда Excel (линейную, полиномиальную, логарифмическую, степенную,
экспоненциальную), выполняет статистическое оценивание и ищет модель с R² >= 75 %.
В папке results создается книга Excel с формулами и диаграммами с линиями тренда.
"""
from pathlib import Path
import json
import math
import sys
import textwrap

import numpy as np
from numpy.polynomial import Polynomial
from scipy import stats

RESULT_DIR = Path(__file__).resolve().parent / "results"
X_NAME, Y_NAME = "Sr_воды", "Sr_волосы"
X_UNIT, Y_UNIT = "мг/л", "мкг/г"
FORECAST_X = 0.424
R2_THRESHOLD = 0.75
ALPHA = 0.05
MAX_DEGREE = 20

# Таблица 1 методички: столбцы Sr_волосы и Sr_воды (29 наблюдений)
SR_HAIR = [4.30, 3.33, 4.45, 6.43, 5.87, 3.90, 5.00, 3.22, 6.15, 3.70, 4.00, 6.04, 7.35,
           10.5, 3.55, 6.43, 9.2, 7.22, 5.98, 3.42, 3.87, 7.41, 4.65, 3.44, 4.66, 8.35,
           4.8, 16.7, 7.22]
SR_WATER = [0.082, 0.088, 0.079, 0.099, 0.101, 0.107, 0.101, 0.115, 0.153, 0.142, 0.116,
            0.117, 0.137, 0.118, 0.128, 0.129, 0.148, 0.152, 0.155, 0.168, 0.168, 0.17,
            0.19, 0.204, 0.207, 0.21, 0.212, 0.222, 0.24]


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


def g(v, digits=6):
    """Число в коротком виде: обычная запись или экспонента для очень больших/малых."""
    if v == 0 or 1e-4 <= abs(v) < 1e7:
        return f"{v:.{digits}g}"
    return f"{v:.3e}"


def round_error(delta):
    """Правило методички: первая значащая цифра <= 3 -> два разряда, иначе один."""
    exponent = math.floor(math.log10(abs(delta)))
    first = int(abs(delta) / 10 ** exponent)
    keep = 2 if first <= 3 else 1
    decimals = -(exponent - keep + 1)
    return round(delta, decimals), max(decimals, 0)


def poly_text(coefs, var="x"):
    """Коэффициенты от старшей степени к младшей -> строка уравнения."""
    deg = len(coefs) - 1
    parts = []
    for k, c in enumerate(coefs):
        p = deg - k
        term = g(abs(c), 5) + (f"·{var}" if p >= 1 else "") + (f"^{p}" if p > 1 else "")
        parts.append(("- " if c < 0 else "+ ") + term)
    text = " ".join(parts)
    return text[2:] if text.startswith("+ ") else "-" + text[2:]


# ---------- ручной расчет по методичке ----------

def manual_linear(x, y):
    n = len(x)
    sx, sy = x.sum(), y.sum()
    sx2, sy2, sxy = (x ** 2).sum(), (y ** 2).sum(), (x * y).sum()
    A = n * sx2 - sx ** 2
    A0 = sy * sx2 - sx * sxy
    A1 = n * sxy - sx * sy
    B = n * sy2 - sy ** 2
    B0 = sx * sy2 - sy * sxy
    B1 = A1
    a0, a1, b0, b1 = A0 / A, A1 / A, B0 / B, B1 / B
    r = A1 / math.sqrt(A * B)
    y_fit = a0 + a1 * x
    r2 = ((y_fit - y.mean()) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    s2y = (sy2 - sy ** 2 / n) / (n - 1)
    s2x = (sx2 - sx ** 2 / n) / (n - 1)
    s_mean_y, s_mean_x = math.sqrt(s2y / n), math.sqrt(s2x / n)
    t = stats.t.ppf(1 - ALPHA / 2, n - 2)
    dy, dx = t * s_mean_y, t * s_mean_x
    dy_r, dy_dec = round_error(dy)
    dx_r, dx_dec = round_error(dx)
    return {
        "n": n, "sum_x": sx, "sum_y": sy, "sum_x2": sx2, "sum_y2": sy2, "sum_xy": sxy,
        "mean_x": sx / n, "mean_y": sy / n,
        "A": A, "A0": A0, "A1": A1, "B": B, "B0": B0, "B1": B1,
        "a0": a0, "a1": a1, "b0": b0, "b1": b1, "r": r, "R2": r2,
        "S2y": s2y, "S2x": s2x, "S_mean_y": s_mean_y, "S_mean_x": s_mean_x,
        "t": t, "dy": dy, "dx": dx, "dy_round": dy_r, "dx_round": dx_r,
        "y_record": f"{sy / n:.{dy_dec}f} ± {dy_r:.{dy_dec}f}",
        "x_record": f"{sx / n:.{dx_dec}f} ± {dx_r:.{dx_dec}f}",
    }


def manual_nonlinear(x, y):
    """Гипербола, экспонента и парабола по формулам (3), (4) и системе методички."""
    n = len(x)

    def line(u, v):
        a1 = (n * (u * v).sum() - u.sum() * v.sum()) / (n * (u ** 2).sum() - u.sum() ** 2)
        return (v.sum() - a1 * u.sum()) / n, a1

    def r2(fit, obs):
        return ((fit - obs.mean()) ** 2).sum() / ((obs - obs.mean()) ** 2).sum()

    z = 1 / x
    h0, h1 = line(z, y)
    b0, b1 = line(x, np.log(y))
    M = np.array([[n, x.sum(), (x ** 2).sum()],
                  [x.sum(), (x ** 2).sum(), (x ** 3).sum()],
                  [(x ** 2).sum(), (x ** 3).sum(), (x ** 4).sum()]])
    rhs = np.array([y.sum(), (x * y).sum(), (x ** 2 * y).sum()])
    p = np.linalg.solve(M, rhs)
    return {
        "hyperbola": {"a0": h0, "a1": h1, "R2": r2(h0 + h1 * z, y),
                      "sum_z": z.sum(), "sum_z2": (z ** 2).sum(), "sum_zy": (z * y).sum()},
        "exponent": {"b0": b0, "b1": b1, "a0": math.exp(b0), "a1": b1,
                     "R2": r2(b0 + b1 * x, np.log(y)),
                     "sum_lny": np.log(y).sum(), "sum_xlny": (x * np.log(y)).sum()},
        "parabola": {"a0": p[0], "a1": p[1], "a2": p[2],
                     "R2": r2(p[0] + p[1] * x + p[2] * x ** 2, y),
                     "system": M.tolist(), "rhs": rhs.tolist()},
    }


# ---------- модели линии тренда Excel ----------

def make_model(kind, x, y, degree=None):
    """Возвращает (коэффициенты, predict, текст уравнения, R² как в Excel)."""
    if kind == "poly":
        c = np.polyfit(x, y, degree)
        predict = lambda t, c=c: np.polyval(c, t)
        return c, predict, "y = " + poly_text(c), None
    if kind == "log":
        a1, a0 = np.polyfit(np.log(x), y, 1)
        predict = lambda t: a0 + a1 * np.log(t)
        return (a0, a1), predict, f"y = {g(a1, 5)}·ln(x) {'+' if a0 >= 0 else '-'} {g(abs(a0), 5)}", None
    if kind == "power":
        a1, ln_a0 = np.polyfit(np.log(x), np.log(y), 1)
        a0 = math.exp(ln_a0)
        predict = lambda t: a0 * t ** a1
        lin_r2 = 1 - ((np.log(y) - (ln_a0 + a1 * np.log(x))) ** 2).sum() / \
            ((np.log(y) - np.log(y).mean()) ** 2).sum()
        return (a0, a1), predict, f"y = {g(a0, 5)}·x^{g(a1, 5)}", lin_r2
    if kind == "exp":
        a1, ln_a0 = np.polyfit(x, np.log(y), 1)
        a0 = math.exp(ln_a0)
        predict = lambda t: a0 * np.exp(a1 * t)
        lin_r2 = 1 - ((np.log(y) - (ln_a0 + a1 * x)) ** 2).sum() / \
            ((np.log(y) - np.log(y).mean()) ** 2).sum()
        return (a0, a1), predict, f"y = {g(a0, 5)}·e^({g(a1, 5)}·x)", lin_r2
    raise ValueError(kind)


MODEL_SPECS = [
    ("Линейная", "poly", 1), ("Полиномиальная 2-й ст.", "poly", 2),
    ("Полиномиальная 3-й ст.", "poly", 3), ("Полиномиальная 4-й ст.", "poly", 4),
    ("Полиномиальная 5-й ст.", "poly", 5), ("Полиномиальная 6-й ст.", "poly", 6),
    ("Логарифмическая", "log", None), ("Степенная", "power", None),
    ("Экспоненциальная", "exp", None),
]


def evaluate_models(x, y):
    n = len(x)
    sst = ((y - y.mean()) ** 2).sum()
    models = []
    for name, kind, degree in MODEL_SPECS:
        coefs, predict, eq, lin_r2 = make_model(kind, x, y, degree)
        fit = predict(x)
        sse = ((y - fit) ** 2).sum()
        r2 = 1 - sse / sst
        k = (degree if kind == "poly" else 1)
        # МНК для степенной и экспоненты выполняется по ln y (R² методички), а диаграмма
        # Excel подписывает квадрат корреляции наблюдаемых y и значений линии тренда
        ln_r2 = lin_r2 if lin_r2 is not None else r2
        excel_r2 = np.corrcoef(y, fit)[0, 1] ** 2 if lin_r2 is not None else r2
        adj = 1 - (1 - r2) * (n - 1) / (n - k - 1)
        F = (ln_r2 / k) / ((1 - ln_r2) / (n - k - 1))
        p = 1 - stats.f.cdf(F, k, n - k - 1)
        loo = []
        for i in range(n):
            keep = np.arange(n) != i
            loo.append(make_model(kind, x[keep], y[keep], degree)[1](x[i:i + 1])[0])
        loo = np.array(loo)
        q2 = 1 - ((y - loo) ** 2).sum() / sst
        models.append({
            "name": name, "kind": kind, "degree": degree, "equation": eq,
            "coefs": [float(c) for c in coefs], "R2_excel": float(excel_r2), "R2_ln": float(ln_r2), "R2": float(r2),
            "R2_adj": float(adj), "rmse": float(math.sqrt(sse / n)), "F": float(F),
            "p": float(p), "Q2_loo": float(q2), "forecast": float(predict(np.array([FORECAST_X]))[0]),
            "predict": predict,
        })
    return models


def linear_inference(x, y):
    n = len(x)
    slope, intercept, r, p, se_slope = stats.linregress(x, y)
    fit = intercept + slope * x
    s = math.sqrt(((y - fit) ** 2).sum() / (n - 2))
    sxx = ((x - x.mean()) ** 2).sum()
    se_int = s * math.sqrt(1 / n + x.mean() ** 2 / sxx)
    t = stats.t.ppf(1 - ALPHA / 2, n - 2)
    F = r ** 2 / (1 - r ** 2) * (n - 2)
    y0 = intercept + slope * FORECAST_X
    se_mean = s * math.sqrt(1 / n + (FORECAST_X - x.mean()) ** 2 / sxx)
    se_pred = s * math.sqrt(1 + 1 / n + (FORECAST_X - x.mean()) ** 2 / sxx)
    spearman = stats.spearmanr(x, y)
    return {
        "a0": intercept, "a1": slope, "se_a0": se_int, "se_a1": se_slope,
        "t_a0": intercept / se_int, "t_a1": slope / se_slope,
        "p_a0": 2 * (1 - stats.t.cdf(abs(intercept / se_int), n - 2)), "p_a1": p,
        "ci_a0": [intercept - t * se_int, intercept + t * se_int],
        "ci_a1": [slope - t * se_slope, slope + t * se_slope],
        "s": s, "F": F, "F_crit": stats.f.ppf(1 - ALPHA, 1, n - 2), "p_F": p,
        "t_crit": t, "forecast": y0,
        "ci_mean": [y0 - t * se_mean, y0 + t * se_mean],
        "ci_pred": [y0 - t * se_pred, y0 + t * se_pred],
        "spearman": float(spearman.statistic), "spearman_p": float(spearman.pvalue),
        "residuals": (y - fit).tolist(), "std_residuals": ((y - fit) / s).tolist(),
    }


def degree_scan(x, y):
    sst = ((y - y.mean()) ** 2).sum()
    rows = []
    for d in range(1, MAX_DEGREE + 1):
        p = Polynomial.fit(x, y, d)
        r2 = 1 - ((y - p(x)) ** 2).sum() / sst
        loo = [Polynomial.fit(np.delete(x, i), np.delete(y, i), d)(x[i]) for i in range(len(x))]
        q2 = 1 - ((y - np.array(loo)) ** 2).sum() / sst
        rows.append({"degree": d, "R2": float(r2), "Q2_loo": float(q2),
                     "forecast": float(p(FORECAST_X))})
    return rows


def best_possible_r2(x, y):
    """Даже функция, проходящая через средние повторных x, не объяснит разброс
    при одинаковых x - это верхняя граница R² для любой зависимости y = f(x)."""
    sst = ((y - y.mean()) ** 2).sum()
    within = sum(((y[x == v] - y[x == v].mean()) ** 2).sum() for v in np.unique(x))
    return 1 - within / sst


# ---------- графики ----------

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#17becf"]


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
        self.plt, self.items = plt, []

    def save(self, fig, name, caption):
        file = f"{len(self.items) + 1:02d}_{name}.png"
        fig.tight_layout()
        fig.savefig(RESULT_DIR / file, bbox_inches="tight")
        self.plt.close(fig)
        self.items.append({"file": file, "caption": caption})


def scatter(ax, x, y, outlier=None):
    ax.scatter(x, y, s=22, color="black", zorder=5, label="эксперимент")
    if outlier is not None:
        ax.scatter(x[outlier], y[outlier], s=90, facecolors="none", edgecolors="red",
                   lw=1.5, zorder=6, label="выброс")
    ax.set_xlabel(f"{X_NAME}, {X_UNIT}")
    ax.set_ylabel(f"{Y_NAME}, {Y_UNIT}")


def make_plots(figs, x, y, models, manual, lin, scan, outlier):
    plt = figs.plt
    grid = np.linspace(x.min(), x.max(), 400)
    pick = [m for m in models if m["name"] in ("Линейная", "Полиномиальная 2-й ст.",
                                               "Логарифмическая", "Степенная", "Экспоненциальная")]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    scatter(ax, x, y, outlier)
    for m, c in zip(pick, COLORS):
        ax.plot(grid, m["predict"](grid), lw=1.6, color=c,
                label=f"{m['name'].lower()}: R² = {m['R2_excel']:.4f}")
    ax.set_title("Экспериментальные точки и пять моделей линии тренда Excel")
    ax.legend()
    figs.save(fig, "models", "Экспериментальные данные и пять видов регрессии")

    show = pick + [m for m in models if m["name"] == "Полиномиальная 6-й ст."]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for ax, m, c in zip(axes.flat, show, COLORS):
        scatter(ax, x, y)
        ax.plot(grid, m["predict"](grid), lw=1.8, color=c)
        ax.set_title(m["name"], fontsize=10)
        eq = textwrap.fill(m["equation"], 34, subsequent_indent="   ")
        ax.text(0.03, 0.97, f"{eq}\nR² = {m['R2_excel']:.4f}", transform=ax.transAxes,
                va="top", fontsize=7.5, bbox=dict(boxstyle="round", fc="white", ec="0.7"))
        ax.set_xlabel(X_NAME)
        ax.set_ylabel(Y_NAME)
    figs.save(fig, "trendlines", "Линии тренда с уравнениями и R², как на диаграммах Excel")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    scatter(ax, x, y)
    ax.plot(grid, manual["a0"] + manual["a1"] * grid, color=COLORS[0], lw=1.8,
            label=f"прямая: y = {manual['a0']:.3f} + {manual['a1']:.3f}x")
    yy = np.linspace(y.min(), y.max(), 200)
    ax.plot(manual["b0"] + manual["b1"] * yy, yy, color=COLORS[1], lw=1.8, ls="--",
            label=f"обратная: x = {manual['b0']:.4f} + {manual['b1']:.5f}y")
    ax.axvline(manual["mean_x"], color="gray", ls=":", lw=1)
    ax.axhline(manual["mean_y"], color="gray", ls=":", lw=1)
    ax.plot(manual["mean_x"], manual["mean_y"], "s", color="black", label="центр (x̄, ȳ)")
    ax.set_title(f"Прямая и обратная регрессия, r = {manual['r']:.4f}")
    ax.legend()
    figs.save(fig, "direct_inverse", "Линии прямой и обратной регрессии")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    z = np.array(lin["std_residuals"])
    ax1.axhspan(-2, 2, color="green", alpha=0.08)
    ax1.axhline(0, color="black", lw=0.8)
    for level in (-3, -2, 2, 3):
        ax1.axhline(level, color="gray", ls=":" if abs(level) == 2 else "--", lw=1)
    colors = ["red" if abs(v) > 3 else COLORS[0] for v in z]
    ax1.scatter(x, z, c=colors, s=24)
    ax1.set_xlabel(X_NAME)
    ax1.set_ylabel("остаток / s")
    ax1.set_title("Нормированные остатки линейной модели")
    stats.probplot(z, dist="norm", plot=ax2)
    ax2.get_lines()[0].set_markerfacecolor(COLORS[0])
    ax2.set_title("Нормальный Q-Q график остатков")
    ax2.set_xlabel("теоретические квантили")
    ax2.set_ylabel("упорядоченные остатки")
    figs.save(fig, "residuals", "Анализ остатков линейной модели")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    d = [r["degree"] for r in scan]
    ax.plot(d, [r["R2"] for r in scan], "-o", color=COLORS[0], ms=4, label="R² на обучающих данных")
    ax.plot(d, [max(r["Q2_loo"], -1.05) for r in scan], "-s", color=COLORS[1], ms=4,
            label="R² скользящего контроля (LOO), обрезано на -1")
    ax.axhline(R2_THRESHOLD, color="black", ls="--", lw=1, label="порог 75 %")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_ylim(-1.1, 1.0)
    ax.set_xticks(d)
    ax.set_xlabel("степень полинома")
    ax.set_ylabel("коэффициент детерминации")
    ax.set_title("Поиск модели с R² ≥ 75 %: рост R² за счет переобучения")
    ax.legend(loc="lower left")
    figs.save(fig, "degree_scan", "Зависимость R² от степени полинома и результат скользящего контроля")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    wide = np.linspace(x.min(), FORECAST_X * 1.04, 500)
    ax.axvspan(x.min(), x.max(), color="gray", alpha=0.12, label="область данных")
    scatter(ax, x, y)
    for m, c in zip(pick, COLORS):
        ax.plot(wide, m["predict"](wide), lw=1.5, color=c, label=m["name"].lower())
        ax.plot(FORECAST_X, m["forecast"], "o", color=c, ms=6)
    ax.errorbar(FORECAST_X, lin["forecast"],
                yerr=[[lin["forecast"] - lin["ci_pred"][0]], [lin["ci_pred"][1] - lin["forecast"]]],
                color=COLORS[0], capsize=6, lw=1.2, label="95 % интервал прогноза (линейная)")
    ax.axvline(FORECAST_X, color="black", ls=":", lw=1)
    ax.set_ylim(0, max(30, lin["ci_pred"][1] * 1.1))
    ax.set_title(f"Экстраполяция моделей до {X_NAME} = {FORECAST_X}")
    ax.legend(fontsize=7.5, loc="upper left")
    figs.save(fig, "forecast", f"Прогноз моделей в точке {X_NAME} = {FORECAST_X}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    power = next(m for m in models if m["kind"] == "power")
    expo = next(m for m in models if m["kind"] == "exp")
    lx, ly = np.log(x), np.log(y)
    ax1.scatter(lx, ly, s=20, color="black")
    ax1.plot(np.log(grid), np.log(power["predict"](grid)), color=COLORS[3])
    ax1.set_title(f"Степенная в координатах (ln x, ln y): R² = {power['R2_ln']:.4f}")
    ax1.set_xlabel("ln x")
    ax1.set_ylabel("ln y")
    ax2.scatter(x, ly, s=20, color="black")
    ax2.plot(grid, np.log(expo["predict"](grid)), color=COLORS[4])
    ax2.set_title(f"Экспонента в координатах (x, ln y): R² = {expo['R2_ln']:.4f}")
    ax2.set_xlabel("x")
    ax2.set_ylabel("ln y")
    figs.save(fig, "linearized", "Линеаризация степенной и экспоненциальной моделей")


# ---------- книга Excel ----------

def build_workbook(path, x, y, models, manual):
    from openpyxl import Workbook
    from openpyxl.chart import Reference, ScatterChart, Series
    from openpyxl.chart.trendline import Trendline
    from openpyxl.chart.shapes import GraphicalProperties
    from openpyxl.drawing.line import LineProperties
    from openpyxl.styles import Alignment, Border, Font, Side

    wb = Workbook()
    head = Font(bold=True)
    thin = Side(style="thin", color="999999")
    box = Border(left=thin, right=thin, top=thin, bottom=thin)
    n = len(x)
    last = n + 1

    ws = wb.active
    ws.title = "Данные"
    ws.append(["№", f"{X_NAME}, {X_UNIT} (x)", f"{Y_NAME}, {Y_UNIT} (y)",
               "x²", "y²", "x·y", "y по линейной модели"])
    for i, (xi, yi) in enumerate(zip(x, y), start=2):
        ws.append([i - 1, float(xi), float(yi), f"=B{i}^2", f"=C{i}^2", f"=B{i}*C{i}",
                   f"='Расчет'!$C$9+'Расчет'!$C$10*B{i}"])
    ws.append(["Σ", f"=SUM(B2:B{last})", f"=SUM(C2:C{last})", f"=SUM(D2:D{last})",
               f"=SUM(E2:E{last})", f"=SUM(F2:F{last})", ""])
    for row in ws.iter_rows(min_row=1, max_row=last + 1, max_col=7):
        for cell in row:
            cell.border = box
            cell.alignment = Alignment(horizontal="center")
    for cell in ws[1] + ws[last + 1]:
        cell.font = head
    for col, width in zip("ABCDEFG", (5, 18, 20, 12, 12, 12, 20)):
        ws.column_dimensions[col].width = width

    rs = wb.create_sheet("Расчет")
    s = f"'Данные'!$B${last + 1}"
    rows = [
        ("Ручной расчет по методичке", None, None),
        ("n", "число наблюдений", f"=COUNT('Данные'!B2:B{last})"),
        ("A", "nΣx² − (Σx)²", f"=C2*'Данные'!D{last + 1}-'Данные'!B{last + 1}^2"),
        ("A0", "ΣyΣx² − ΣxΣxy", f"='Данные'!C{last + 1}*'Данные'!D{last + 1}-{s}*'Данные'!F{last + 1}"),
        ("A1", "nΣxy − ΣxΣy", f"=C2*'Данные'!F{last + 1}-{s}*'Данные'!C{last + 1}"),
        ("B", "nΣy² − (Σy)²", f"=C2*'Данные'!E{last + 1}-'Данные'!C{last + 1}^2"),
        ("B0", "ΣxΣy² − ΣyΣxy", f"={s}*'Данные'!E{last + 1}-'Данные'!C{last + 1}*'Данные'!F{last + 1}"),
        ("B1", "равно A1", "=C5"),
        ("a0", "A0 / A", "=C4/C3"),
        ("a1", "A1 / A", "=C5/C3"),
        ("b0", "B0 / B (обратная регрессия)", "=C7/C6"),
        ("b1", "B1 / B", "=C8/C6"),
        ("r", "A1 / √(A·B)", "=C5/SQRT(C3*C6)"),
        ("R²", "r² (для линейной модели)", "=C13^2"),
        ("S²y", "несмещенная дисперсия y", f"=('Данные'!E{last + 1}-'Данные'!C{last + 1}^2/C2)/(C2-1)"),
        ("S²x", "несмещенная дисперсия x", f"=('Данные'!D{last + 1}-{s}^2/C2)/(C2-1)"),
        ("Sȳ", "√(S²y / n)", "=SQRT(C15/C2)"),
        ("Sx̄", "√(S²x / n)", "=SQRT(C16/C2)"),
        ("t", f"квантиль Стьюдента, α={ALPHA}, ν=n−2", "=TINV(0.05,C2-2)"),
        ("Δy", "t·Sȳ", "=C19*C17"),
        ("Δx", "t·Sx̄", "=C19*C18"),
        ("ȳ", "среднее y", f"=AVERAGE('Данные'!C2:C{last})"),
        ("x̄", "среднее x", f"=AVERAGE('Данные'!B2:B{last})"),
        ("Проверка встроенными функциями Excel", None, None),
        ("a1", "НАКЛОН", f"=SLOPE('Данные'!C2:C{last},'Данные'!B2:B{last})"),
        ("a0", "ОТРЕЗОК", f"=INTERCEPT('Данные'!C2:C{last},'Данные'!B2:B{last})"),
        ("r", "КОРРЕЛ", f"=CORREL('Данные'!B2:B{last},'Данные'!C2:C{last})"),
        ("R²", "КВПИРСОН", f"=RSQ('Данные'!C2:C{last},'Данные'!B2:B{last})"),
        ("Прогноз", None, None),
        ("x*", "заданное значение", FORECAST_X),
        ("y*", "a0 + a1·x*", "=C9+C10*C30"),
        ("Вывод", "R² ≥ 75 %?", '=IF(C14>=0.75,"прогноз допустим","прогноз недопустим")'),
    ]
    for i, (a, b, c) in enumerate(rows, start=1):
        rs.cell(i, 1, a)
        if b is None:
            rs.cell(i, 1).font = Font(bold=True, size=12)
            continue
        rs.cell(i, 2, b)
        rs.cell(i, 3, c)
        rs.cell(i, 1).font = head
        for col in (1, 2, 3):
            rs.cell(i, col).border = box
        rs.cell(i, 3).number_format = "0.000000"
    rs.cell(2, 3).number_format = "0"
    rs.column_dimensions["A"].width = 12
    rs.column_dimensions["B"].width = 38
    rs.column_dimensions["C"].width = 18

    ms = wb.create_sheet("Модели")
    ms.append(["Модель", "Уравнение", "R² (как в Excel)", "R² по ln y", "R² по исходным y",
               "R² скорр.", "RMSE", "F", "p-значение", "R² LOO", f"Прогноз при x = {FORECAST_X}"])
    for m in models:
        ms.append([m["name"], m["equation"], m["R2_excel"], m["R2_ln"], m["R2"], m["R2_adj"],
                   m["rmse"], m["F"], m["p"], m["Q2_loo"], m["forecast"]])
    for cell in ms[1]:
        cell.font = head
        cell.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
    for col, width in zip("ABCDEFGHIJK", (23, 46, 10, 10, 10, 9, 8, 8, 10, 10, 13)):
        ms.column_dimensions[col].width = width
    for row in ms.iter_rows(min_row=1, max_row=len(models) + 1, max_col=11):
        for cell in row:
            cell.border = box
            if cell.row > 1:
                cell.alignment = Alignment(wrap_text=True, vertical="center")
                if cell.column >= 3:
                    cell.number_format = "0.0000"

    cs = wb.create_sheet("Диаграммы")
    xs = Reference(ws, min_col=2, min_row=2, max_row=last)
    ys = Reference(ws, min_col=3, min_row=2, max_row=last)
    trends = [("Линейная", dict(trendlineType="linear")),
              ("Полиномиальная (2-я степень)", dict(trendlineType="poly", order=2)),
              ("Логарифмическая", dict(trendlineType="log")),
              ("Степенная", dict(trendlineType="power")),
              ("Экспоненциальная", dict(trendlineType="exp")),
              ("Полиномиальная (6-я степень)", dict(trendlineType="poly", order=6))]
    for k, (title, params) in enumerate(trends):
        chart = ScatterChart()
        chart.title = f"{title} регрессия"
        chart.x_axis.title = f"{X_NAME}, {X_UNIT}"
        chart.y_axis.title = f"{Y_NAME}, {Y_UNIT}"
        for item in (chart.title, chart.x_axis.title, chart.y_axis.title):
            item.overlay = False
        chart.x_axis.scaling.min = 0.05
        chart.x_axis.scaling.max = 0.25
        chart.x_axis.majorUnit = 0.05
        chart.x_axis.number_format = "0.00"
        chart.x_axis.delete = False
        chart.y_axis.delete = False
        chart.legend = None
        series = Series(ys, xs, title="эксперимент")
        series.marker.symbol = "circle"
        series.marker.size = 6
        series.marker.graphicalProperties = GraphicalProperties(solidFill="1F77B4")
        series.marker.graphicalProperties.line.solidFill = "1F77B4"
        series.graphicalProperties.line.noFill = True
        series.trendline = Trendline(dispRSqr=True, dispEq=True, **params)
        series.trendline.spPr = GraphicalProperties(ln=LineProperties(solidFill="D62728", w=22000))
        chart.series.append(series)
        chart.width, chart.height = 16, 9
        cs.add_chart(chart, f"{'AJ'[k % 2]}{1 + (k // 2) * 19}")
    wb.save(path)


def pdf_to_png(pdf_path, scale=2.5):
    """Страница PDF, сохраненная Excel, -> PNG без пустых полей."""
    import pypdfium2 as pdfium
    from PIL import ImageChops
    pdf = pdfium.PdfDocument(str(pdf_path))
    image = pdf[0].render(scale=scale).to_pil().convert("RGB")
    pdf.close()
    background = image.copy()
    background.paste((255, 255, 255), (0, 0, *image.size))
    box = ImageChops.difference(image, background).getbbox()
    if box:
        pad = 12
        image = image.crop((max(box[0] - pad, 0), max(box[1] - pad, 0),
                            min(box[2] + pad, image.width), min(box[3] + pad, image.height)))
    png = pdf_path.with_suffix(".png")
    image.save(png)
    pdf_path.unlink()
    return png.name


def excel_export(xlsx, out):
    """Если установлен Microsoft Excel: пересчитать книгу в самом Excel, сохранить
    картинки листов и диаграмм и сверить R² на диаграммах с расчетом Python."""
    script = Path(__file__).with_name("excel_export.ps1")
    if sys.platform != "win32" or not script.exists():
        out("  Excel недоступен - шаг пропущен")
        return None
    import subprocess
    try:
        run = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                              str(script), "-Workbook", str(xlsx), "-OutDir", str(RESULT_DIR)],
                             capture_output=True, timeout=300)
        text = run.stdout.decode("utf-8", errors="replace").strip().splitlines()
        result = json.loads(text[-1]) if run.returncode == 0 and text else None
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        result = None
    if not result:
        out("  Не удалось открыть книгу в Excel - шаг пропущен (на расчеты не влияет)")
        return None
    result["ranges"] = [pdf_to_png(RESULT_DIR / name) for name in result["ranges"]]
    for chart in result["charts"]:
        out(f"  {chart['title']}: {chart['label'] or '(подпись есть на картинке диаграммы)'}")
    out("  Картинки: " + ", ".join(result["ranges"] + [c["file"] for c in result["charts"]]))
    return result


# ---------- главная программа ----------

def main():
    RESULT_DIR.mkdir(exist_ok=True)
    for old in [*RESULT_DIR.glob("*.png"), *RESULT_DIR.glob("*.pdf")]:
        old.unlink()
    out = Tee(RESULT_DIR / "results.txt")
    x, y = np.array(SR_WATER), np.array(SR_HAIR)
    n = len(x)
    out("Лабораторная работа 4. Построение аналитической модели по результатам эксперимента")
    out(f"Вариант 12, задание 12: {Y_NAME} = f({X_NAME}), прогноз при {X_NAME} = {FORECAST_X}")
    out(f"Python {sys.version.split()[0]}, numpy {np.__version__}\n")

    out("1. ИСХОДНЫЕ ДАННЫЕ (таблица 1 методички) и промежуточные величины")
    rows = [[i + 1, g(xi, 4), g(yi, 4), f"{xi ** 2:.6f}", f"{yi ** 2:.4f}", f"{xi * yi:.5f}"]
            for i, (xi, yi) in enumerate(zip(x, y))]
    man = manual_linear(x, y)
    rows.append(["Σ", f"{man['sum_x']:.3f}", f"{man['sum_y']:.2f}", f"{man['sum_x2']:.6f}",
                 f"{man['sum_y2']:.4f}", f"{man['sum_xy']:.5f}"])
    out(ascii_table(["i", "x", "y", "x²", "y²", "x·y"], rows))
    desc = {"x": stats.describe(x), "y": stats.describe(y)}
    out(ascii_table(["величина", "min", "max", "среднее", "медиана", "ст. откл."],
                    [[k, g(d.minmax[0], 4), g(d.minmax[1], 4), g(d.mean, 5),
                      g(float(np.median(v)), 4), g(math.sqrt(d.variance), 5)]
                     for (k, d), v in zip(desc.items(), (x, y))]))

    out("\n2. ЛИНЕЙНАЯ РЕГРЕССИЯ: РУЧНОЙ РАСЧЕТ ПО ФОРМУЛАМ МЕТОДИЧКИ")
    for key, text in [("A", "nΣx² − (Σx)²"), ("A0", "ΣyΣx² − ΣxΣxy"), ("A1", "nΣxy − ΣxΣy"),
                      ("B", "nΣy² − (Σy)²"), ("B0", "ΣxΣy² − ΣyΣxy"), ("B1", "= A1")]:
        out(f"  {key:3} = {text:16} = {man[key]:.6f}")
    out(f"  a0 = A0/A = {man['a0']:.4f};  a1 = A1/A = {man['a1']:.4f}")
    out(f"  b0 = B0/B = {man['b0']:.5f};  b1 = B1/B = {man['b1']:.6f}")
    out(f"  Прямая регрессия:   y = {man['a0']:.3f} + {man['a1']:.3f}·x")
    out(f"  Обратная регрессия: x = {man['b0']:.4f} + {man['b1']:.5f}·y")
    r = man["r"]
    strength = ("практически отсутствует" if abs(r) < 0.3 else "слабая" if abs(r) < 0.5
                else "умеренная" if abs(r) < 0.7 else "сильная")
    out(f"  r = A1/√(A·B) = {r:.4f} -> связь вероятностная, {'прямая' if r > 0 else 'обратная'}, {strength}")
    out(f"  R² = {man['R2']:.4f} ({man['R2'] * 100:.2f} %)")
    out(f"  S²y = {man['S2y']:.4f},  S²x = {man['S2x']:.7f}")
    out(f"  Sȳ = {man['S_mean_y']:.4f},  Sx̄ = {man['S_mean_x']:.6f}")
    out(f"  t(α={ALPHA}; ν={n - 2}) = {man['t']:.4f}")
    out(f"  Δy = {man['dy']:.4f} -> {man['dy_round']};  Δx = {man['dx']:.6f} -> {man['dx_round']}")
    out(f"  Запись результата: y = {man['y_record']} {Y_UNIT};  x = {man['x_record']} {X_UNIT}")
    allowed = man["R2"] >= R2_THRESHOLD
    out(f"  R² = {man['R2'] * 100:.2f} % {'>=' if allowed else '<'} 75 % -> прогноз по линейной "
        f"модели {'допустим' if allowed else 'НЕДОПУСТИМ'}")

    out("\n3. НЕЛИНЕЙНЫЕ РЕГРЕССИИ ПО ФОРМУЛАМ МЕТОДИЧКИ")
    nl = manual_nonlinear(x, y)
    h, e, p = nl["hyperbola"], nl["exponent"], nl["parabola"]
    out(f"  Гипербола  y = a0 + a1/x:   Σz = {h['sum_z']:.4f}, Σz² = {h['sum_z2']:.4f}, "
        f"Σzy = {h['sum_zy']:.4f}")
    out(f"             y = {h['a0']:.4f} {'+' if h['a1'] >= 0 else '-'} {abs(h['a1']):.5f}/x,  R² = {h['R2']:.4f}")
    out(f"  Экспонента ln y = b0 + b1·x: Σln y = {e['sum_lny']:.4f}, Σx·ln y = {e['sum_xlny']:.5f}")
    out(f"             b0 = {e['b0']:.5f}, b1 = {e['b1']:.5f} -> y = {e['a0']:.4f}·e^({e['a1']:.4f}x),"
        f"  R²(ln y) = {e['R2']:.4f}")
    out("  Парабола   система нормальных уравнений:")
    for row_, rhs in zip(p["system"], p["rhs"]):
        out("             [" + "  ".join(f"{v:12.6f}" for v in row_) + f" ] · a = {rhs:.6f}")
    out(f"             y = {p['a0']:.4f} {'+' if p['a1'] >= 0 else '-'} {abs(p['a1']):.4f}x "
        f"{'+' if p['a2'] >= 0 else '-'} {abs(p['a2']):.4f}x²,  R² = {p['R2']:.4f}")

    out("\n4. МОДЕЛИ ЛИНИИ ТРЕНДА EXCEL")
    out("  R² Excel - значение на диаграмме; R² (ln y) - по методичке для степенной и экспоненты;")
    out("  R² по y = 1 - SSE/SST в исходных единицах; R² LOO - скользящий контроль")
    models = evaluate_models(x, y)
    out(ascii_table(["модель", "R² Excel", "R² (ln y)", "R² по y", "R² скорр.", "RMSE",
                     "p (F-тест)", "R² LOO", f"y({FORECAST_X})"],
                    [[m["name"], f"{m['R2_excel']:.4f}",
                      f"{m['R2_ln']:.4f}" if m["kind"] in ("power", "exp") else "-",
                      f"{m['R2']:.4f}", f"{m['R2_adj']:.4f}",
                      f"{m['rmse']:.3f}", f"{m['p']:.4f}", f"{m['Q2_loo']:.3f}", g(m["forecast"], 5)]
                     for m in models]))
    for m in models:
        out(f"  {m['name']:24} {m['equation']}")
    excel5 = [m for m in models if m["name"] in ("Линейная", "Полиномиальная 2-й ст.",
                                                 "Логарифмическая", "Степенная", "Экспоненциальная")]
    best = max(excel5, key=lambda m: m["R2_excel"])
    out(f"  Лучшая из пяти обязательных моделей по R²: {best['name']} (R² = {best['R2_excel']:.4f})")

    out("\n5. СТАТИСТИЧЕСКОЕ ОЦЕНИВАНИЕ ЛИНЕЙНОЙ МОДЕЛИ")
    lin = linear_inference(x, y)
    out(ascii_table(["коэфф.", "оценка", "ст. ошибка", "t", "p", "95 % интервал"],
                    [["a0", f"{lin['a0']:.4f}", f"{lin['se_a0']:.4f}", f"{lin['t_a0']:.3f}",
                      f"{lin['p_a0']:.4f}", f"[{lin['ci_a0'][0]:.3f}; {lin['ci_a0'][1]:.3f}]"],
                     ["a1", f"{lin['a1']:.4f}", f"{lin['se_a1']:.4f}", f"{lin['t_a1']:.3f}",
                      f"{lin['p_a1']:.4f}", f"[{lin['ci_a1'][0]:.3f}; {lin['ci_a1'][1]:.3f}]"]]))
    out(f"  F = {lin['F']:.4f}, F крит(0,05; 1; {n - 2}) = {lin['F_crit']:.4f}, p = {lin['p_F']:.4f}")
    out(f"  Стандартная ошибка оценки s = {lin['s']:.4f} {Y_UNIT}")
    out(f"  Ранговая корреляция Спирмена ρ = {lin['spearman']:.4f}, p = {lin['spearman_p']:.4f}")
    z = np.array(lin["std_residuals"])
    outlier = int(np.argmax(np.abs(z)))
    out(f"  Наибольший нормированный остаток: наблюдение №{outlier + 1} "
        f"(x = {x[outlier]}, y = {y[outlier]}), остаток = {z[outlier]:.2f}·s")
    keep = np.arange(n) != outlier
    no_out = stats.linregress(x[keep], y[keep])
    out(f"  Без наблюдения №{outlier + 1}: y = {no_out.intercept:.3f} + {no_out.slope:.3f}x, "
        f"R² = {no_out.rvalue ** 2:.4f}, p = {no_out.pvalue:.3f}")
    out(f"  Прогноз по линейной модели (только для иллюстрации): y({FORECAST_X}) = {lin['forecast']:.3f}")
    out(f"    95 % интервал среднего:  [{lin['ci_mean'][0]:.3f}; {lin['ci_mean'][1]:.3f}]")
    out(f"    95 % интервал прогноза: [{lin['ci_pred'][0]:.3f}; {lin['ci_pred'][1]:.3f}]")
    out(f"  x* = {FORECAST_X} лежит вне диапазона данных [{x.min()}; {x.max()}] "
        f"({FORECAST_X / x.max():.2f}·xmax) -> это экстраполяция")

    out("\n6. ПОИСК МОДЕЛИ С R² >= 75 %")
    scan = degree_scan(x, y)
    out(ascii_table(["степень", "R²", "R² LOO", f"прогноз y({FORECAST_X})"],
                    [[r_["degree"], f"{r_['R2']:.4f}", g(r_["Q2_loo"], 4), g(r_["forecast"], 4)]
                     for r_ in scan]))
    first = next((r_ for r_ in scan if r_["R2"] >= R2_THRESHOLD), None)
    if first:
        out(f"  Формально R² >= 75 % впервые при степени {first['degree']}: R² = {first['R2']:.4f}, "
            f"но прогноз {g(first['forecast'], 4)} и R² LOO = {g(first['Q2_loo'], 4)} -> переобучение")
    upper = best_possible_r2(x, y)
    dups = [float(v) for v in np.unique(x) if (x == v).sum() > 1]
    out(f"  Повторяющиеся значения x: {dups}; верхняя граница R² для любой функции y = f(x): {upper:.4f}")
    q2_best = max(scan, key=lambda r_: r_["Q2_loo"])
    out(f"  Наибольший R² скользящего контроля: {q2_best['Q2_loo']:.4f} (степень {q2_best['degree']}) "
        "-> ни одна модель не предсказывает новые наблюдения лучше среднего значения")

    out("\n7. ВЫВОД")
    out(f"  Связь {Y_NAME} и {X_NAME} слабая: r = {r:.3f}, R² не превышает {best['R2_excel'] * 100:.1f} % "
        "для обязательных моделей.")
    out("  Прогноз при Sr_воды = 0,424 по найденным моделям выполнять нельзя.")

    plt = setup_matplotlib()
    figs = Figures(plt)
    make_plots(figs, x, y, models, man, lin, scan, outlier)
    xlsx = RESULT_DIR / "lab4_excel.xlsx"
    build_workbook(xlsx, x, y, models, man)
    out(f"\nГрафики: {', '.join(f['file'] for f in figs.items)}")
    out(f"Книга Excel: {xlsx.name} (формулы ручного расчета и 6 диаграмм с линиями тренда)")
    out("\n8. ПРОВЕРКА В MICROSOFT EXCEL (подписи линий тренда, рассчитанные самим Excel)")
    excel = excel_export(xlsx, out)

    for m in models:
        m.pop("predict")
    data = {"x": x.tolist(), "y": y.tolist(), "forecast_x": FORECAST_X, "manual": man,
            "nonlinear": nl, "models": models, "best": best["name"], "linear": lin,
            "outlier": {"index": outlier + 1, "x": float(x[outlier]), "y": float(y[outlier]),
                        "z": float(z[outlier]), "r2_without": float(no_out.rvalue ** 2),
                        "a0_without": float(no_out.intercept), "a1_without": float(no_out.slope),
                        "p_without": float(no_out.pvalue)},
            "scan": scan, "first_degree": first, "upper_r2": upper, "duplicates": dups,
            "q2_best": q2_best,
            "describe": {k: {"min": float(d.minmax[0]), "max": float(d.minmax[1]),
                             "mean": float(d.mean), "std": float(math.sqrt(d.variance)),
                             "median": float(np.median(v))}
                         for (k, d), v in zip(desc.items(), (x, y))},
            "figures": figs.items, "excel": excel}
    (RESULT_DIR / "results.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    out(f"\nРезультаты сохранены в папке {RESULT_DIR}")
    out.file.close()


if __name__ == "__main__":
    main()

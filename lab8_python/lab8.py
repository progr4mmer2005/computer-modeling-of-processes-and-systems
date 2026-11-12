from itertools import product
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

folder = Path(__file__).parent / "results"
folder.mkdir(exist_ok=True)


def smf(x, a, b):
    x = np.asarray(x, dtype=float)
    m = (a + b) / 2
    y = np.where(x <= m, 2 * ((x - a) / (b - a)) ** 2, 1 - 2 * ((x - b) / (b - a)) ** 2)
    return np.where(x <= a, 0.0, np.where(x >= b, 1.0, y))


def zmf(x, a, b):
    return 1 - smf(x, a, b)


def pimf(x, a, b, c, d):
    return np.minimum(smf(x, a, b), zmf(x, c, d))


def line(x, points, values):
    return np.interp(x, points, values)


def centroid(u, mu):
    return (u * mu).sum() / mu.sum() if mu.sum() > 0 else np.nan


print("ЛАБОРАТОРНАЯ РАБОТА 8. СИСТЕМА НЕЧЁТКОГО ВЫВОДА МАМДАНИ")
print("Вариант 12 -> задание 4: функции принадлежности выхода, вариант 4")
print("Операции: И = min, импликация = min, агрегирование = max, дефаззификация = центр тяжести")

print("\n1. ЭКОСИСТЕМА: 4 ВХОДА, 4 ТЕРМА, 256 ПРАВИЛ")
names = ["P-air", "P-snow", "P-ground", "P-bio"]
terms = ["отличное", "хорошее", "напряжённое", "плохое"]
params = {
    "узкие": [(0.0, 0.35), (0.329, 0.392, 0.4034, 0.469), (0.47, 0.6, 0.65, 0.8), (0.6, 1.0)],
    "базовые": [(0.0, 0.4), (0.1, 0.3, 0.4, 0.6), (0.4, 0.6, 0.7, 0.9), (0.6, 1.0)],
    "широкие": [(0.0, 0.5), (0.0, 0.25, 0.45, 0.7), (0.3, 0.55, 0.75, 1.0), (0.5, 1.0)],
}
print("Функции принадлежности термов (одинаковые для входов и выхода, диапазон [0; 1]):")
for key, p in params.items():
    print(f"  {key:8} отличное zmf{p[0]}  хорошее pimf{p[1]}  напряжённое pimf{p[2]}  плохое smf{p[3]}")
rules = [(combo, int(np.floor(np.mean(combo) + 0.5))) for combo in product(range(4), repeat=4)]
print("Правило: итоговое состояние = среднее номеров термов входов, округлённое")
print("(0 - отличное, 1 - хорошее, 2 - напряжённое, 3 - плохое). Первые правила:")
for combo, out in rules[:6]:
    text = " И ".join(f"{n} {terms[t]}" for n, t in zip(names, combo))
    print(f"  ЕСЛИ {text} ТО состояние {terms[out]}")
print(f"  ... всего правил: {len(rules)}")
count = [sum(1 for _, o in rules if o == k) for k in range(4)]
print("Правил по выходным термам:", ", ".join(f"{t} {c}" for t, c in zip(terms, count)))

u = np.linspace(0, 1, 501)


def eco_terms(x, p):
    return [zmf(x, *p[0]), pimf(x, *p[1]), pimf(x, *p[2]), smf(x, *p[3])]


def eco(inputs, p):
    mu = [eco_terms(v, p) for v in inputs]
    strength = np.zeros(4)
    for combo, out in rules:
        w = min(mu[i][t] for i, t in enumerate(combo))
        strength[out] = max(strength[out], w)
    out_mu = eco_terms(u, p)
    agg = np.max([np.minimum(strength[k], out_mu[k]) for k in range(4)], axis=0)
    return centroid(u, agg), strength


x0 = [0.5, 0.5, 0.5, 0.5]
print("\nВход [0,5 0,5 0,5 0,5], базовые функции:")
mu_air = [float(m) for m in eco_terms(0.5, params["базовые"])]
print("  степени принадлежности 0,5:", ", ".join(f"{t} {m:.3f}" for t, m in zip(terms, mu_air)))
value, strength = eco(x0, params["базовые"])
print("  сила выходных термов:", ", ".join(f"{t} {s:.3f}" for t, s in zip(terms, strength)))
print(f"  состояние экосистемы = {value:.3f}")

print("\nЗависимость выхода от параметров функций принадлежности:")
tests = [[0.2] * 4, [0.5] * 4, [0.8] * 4, [0.1, 0.1, 0.9, 0.9], [0.3, 0.5, 0.6, 0.4]]
print(f"  {'вход':24}" + "".join(f"{k:>10}" for k in params))
eco_table = {}
for x in tests:
    row = [eco(x, p)[0] for p in params.values()]
    eco_table[tuple(x)] = row
    label = "[" + " ".join(f"{v:.1f}" for v in x) + "]"
    print(f"  {label:24}" + "".join(f"{v:10.3f}" for v in row))

sweep = np.linspace(0, 1, 41)
eco_curves = {k: [eco([a, 0.5, 0.5, 0.5], p)[0] for a in sweep] for k, p in params.items()}
for k, c in eco_curves.items():
    print(f"  {k:8}: P-air 0 -> 1 при остальных 0,5 даёт выход {min(c):.3f} ... {max(c):.3f}")

grid = np.linspace(0, 1, 21)
eco_surface = np.array([[eco([a, s, 0.5, 0.5], params["базовые"])[0] for a in grid] for s in grid])

print("\n2. КОНДИЦИОНЕР MITSUBISHI HEAVY SRK25ZJP-S")
dt_terms = {"малая": ([0, 15], [1, 0]), "средняя": ([0, 15, 30], [0, 1, 0]), "большая": ([15, 30], [0, 1])}
vt_terms = {"малая": ([0, 0.15], [1, 0]), "средняя": ([0, 0.15, 0.3], [0, 1, 0]), "большая": ([0.15, 0.3], [0, 1])}
out_names = ["очень малая", "малая", "средняя", "большая", "очень большая"]
peaks = {1: [1.0, 37, 62, 87, 115], 2: [0.5, 32, 64, 96, 128], 3: [3, 33, 63, 93, 123], 4: [1.5, 35, 68, 101, 134]}
variant = 4
f = np.linspace(0, 150, 1501)


def out_terms(p):
    return {
        "очень малая": ([p[0], p[1]], [1, 0]),
        "малая": ([p[0], p[1], p[2]], [0, 1, 0]),
        "средняя": ([p[1], p[2], p[3]], [0, 1, 0]),
        "большая": ([p[2], p[3], p[4]], [0, 1, 0]),
        "очень большая": ([p[3], p[4]], [0, 1]),
    }


ac_rules = [
    ("малая", "малая", "очень малая"), ("малая", "средняя", "малая"), ("малая", "большая", "средняя"),
    ("средняя", "малая", "малая"), ("средняя", "средняя", "средняя"), ("средняя", "большая", "большая"),
    ("большая", "малая", "средняя"), ("большая", "средняя", "большая"), ("большая", "большая", "очень большая"),
]
print("Вход 1: разность температур Δt, К (0...35): малая, средняя, большая (вершины 0, 15, 30)")
print("Вход 2: скорость изменения температуры Vt, К/мин (0...0,35): вершины 0, 0,15, 0,3")
print(f"Выход: частота вращения компрессора f, Гц (0...150), вариант {variant}: вершины "
      + ", ".join(f"{n} {v:g}" for n, v in zip(out_names, peaks[variant])))
print("Правила (скорость, разность -> частота):")
for k, (v, d, o) in enumerate(ac_rules, 1):
    print(f"  {k}. ЕСЛИ Vt {v} И Δt {d} ТО f {o}")


def conditioner(dt, vt, p):
    terms_f = out_terms(p)
    fired = []
    agg = np.zeros_like(f)
    for v, d, o in ac_rules:
        w = min(line(vt, *vt_terms[v]), line(dt, *dt_terms[d]))
        fired.append(w)
        agg = np.maximum(agg, np.minimum(w, line(f, *terms_f[o])))
    return centroid(f, agg), fired, agg


dt0, vt0 = 20.0, 0.1
print(f"\nПример: Δt = {dt0:g} К, Vt = {vt0:g} К/мин")
for name, t in dt_terms.items():
    print(f"  μ_{name}(Δt) = {line(dt0, *t):.3f}", end="")
print()
for name, t in vt_terms.items():
    print(f"  μ_{name}(Vt) = {line(vt0, *t):.3f}", end="")
print()
f0, fired0, agg0 = conditioner(dt0, vt0, peaks[variant])
for k, ((v, d, o), w) in enumerate(zip(ac_rules, fired0), 1):
    if w > 0:
        print(f"  правило {k}: min(μ_{v}(Vt), μ_{d}(Δt)) = {w:.3f} -> f {o}")
print(f"  центр тяжести: f = Σ f·μ(f) / Σ μ(f) = {f0:.2f} Гц")

dts = np.arange(0, 36, 5)
vts = np.round(np.arange(0, 0.351, 0.05), 2)
ac_table = np.array([[conditioner(d, v, peaks[variant])[0] for d in dts] for v in vts])
print("\nЧастота f, Гц (строки - Vt, столбцы - Δt):")
print("  Vt \\ Δt" + "".join(f"{d:8g}" for d in dts))
for v, row in zip(vts, ac_table):
    print(f"  {v:7.2f} " + "".join(f"{x:8.1f}" for x in row))
print(f"Наименьшая частота {ac_table.min():.1f} Гц, наибольшая {ac_table.max():.1f} Гц")

print("\n3. СРАВНЕНИЕ ВАРИАНТОВ ФУНКЦИЙ ПРИНАДЛЕЖНОСТИ ВЫХОДА")
points = [(0, 0), (10, 0.05), (20, 0.1), (30, 0.3)]
print("  вариант   вершины термов, Гц            " + "".join(f"  Δt={d:g},Vt={v:g}" for d, v in points) + "   min    max")
variant_rows = {}
for k, p in peaks.items():
    vals = [conditioner(d, v, p)[0] for d, v in points]
    full = np.array([[conditioner(d, v, p)[0] for d in dts] for v in vts])
    variant_rows[k] = (vals, full.min(), full.max())
    mark = " <- наш" if k == variant else ""
    print(f"  {k:7}   {str(p):30}" + "".join(f"{x:16.1f}" for x in vals) + f"{full.min():7.1f}{full.max():7.1f}{mark}")
print(f"Вариант {variant}: шаг между вершинами термов около {np.diff(peaks[variant][1:]).mean():.0f} Гц, "
      f"диапазон выхода {ac_table.min():.1f} ... {ac_table.max():.1f} Гц")

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for ax, (title, p) in zip(axes.flat, list(params.items()) + [("выход (базовые)", params["базовые"])]):
    for t, m in zip(terms, eco_terms(u, p)):
        ax.plot(u, m, label=t)
    ax.set_title(f"Функции принадлежности: {title}")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
axes[0, 0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(folder / "01_eco_membership.png", dpi=130)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4.5))
for k, c in eco_curves.items():
    ax.plot(sweep, c, label=k)
ax.set_xlabel("P-air (остальные входы 0,5)")
ax.set_ylabel("Состояние экосистемы")
ax.legend(title="функции принадлежности")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(folder / "02_eco_params.png", dpi=130)
plt.close(fig)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(projection="3d")
A, S = np.meshgrid(grid, grid)
ax.plot_surface(A, S, eco_surface, cmap="viridis")
ax.set_xlabel("P-air")
ax.set_ylabel("P-snow")
ax.set_zlabel("состояние")
ax.set_title("Поверхность вывода (P-ground = P-bio = 0,5)")
fig.tight_layout()
fig.savefig(folder / "03_eco_surface.png", dpi=130)
plt.close(fig)

fig, axes = plt.subplots(3, 1, figsize=(9, 9))
xs = np.linspace(0, 35, 351)
for name, t in dt_terms.items():
    axes[0].plot(xs, line(xs, *t), label=name)
axes[0].set_title("Разность температур Δt, К")
xs = np.linspace(0, 0.35, 351)
for name, t in vt_terms.items():
    axes[1].plot(xs, line(xs, *t), label=name)
axes[1].set_title("Скорость изменения температуры Vt, К/мин")
for name, t in out_terms(peaks[variant]).items():
    axes[2].plot(f, line(f, *t), label=name)
for v in peaks[variant]:
    axes[2].annotate(f"{v:g}", (v, 1.02), ha="center", fontsize=8)
axes[2].set_title(f"Частота вращения компрессора f, Гц (вариант {variant})")
for ax in axes:
    ax.set_ylim(-0.05, 1.12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="center right")
fig.tight_layout()
fig.savefig(folder / "04_ac_membership.png", dpi=130)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 4))
for name, t in out_terms(peaks[variant]).items():
    ax.plot(f, line(f, *t), color="gray", lw=0.8)
ax.fill_between(f, agg0, alpha=0.5, label="агрегированный выход")
ax.axvline(f0, color="red", label=f"центр тяжести f = {f0:.1f} Гц")
ax.set_title(f"Вывод для Δt = {dt0:g} К, Vt = {vt0:g} К/мин")
ax.set_xlabel("f, Гц")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(folder / "05_ac_example.png", dpi=130)
plt.close(fig)

D, V = np.meshgrid(np.linspace(0, 35, 36), np.linspace(0, 0.35, 36))
Z = np.vectorize(lambda d, v: conditioner(d, v, peaks[variant])[0])(D, V)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(projection="3d")
ax.plot_surface(D, V, Z, cmap="viridis")
ax.set_xlabel("Δt, К")
ax.set_ylabel("Vt, К/мин")
ax.set_zlabel("f, Гц")
ax.set_title(f"Поверхность вывода кондиционера (вариант {variant})")
fig.tight_layout()
fig.savefig(folder / "06_ac_surface.png", dpi=130)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4.5))
for k, p in peaks.items():
    ax.plot(xs * 100, [conditioner(20, v, p)[0] for v in xs], lw=2.5 if k == variant else 1,
            label=f"вариант {k}" + (" (наш)" if k == variant else ""))
ax.set_xlabel("Vt, К/мин × 100 (Δt = 20 К)")
ax.set_ylabel("f, Гц")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(folder / "07_variants.png", dpi=130)
plt.close(fig)

print("\nГрафики сохранены в папку results")

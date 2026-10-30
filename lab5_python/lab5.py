from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit

folder = Path(__file__).parent / "results"
folder.mkdir(exist_ok=True)

x = np.array([0.0012, 0.0015, 0.0062, 0.0013, 0.0042, 0.0019, 0.0012, 0.0014, 0.0064, 0.0013,
              0.001, 0.0061, 0.0011, 0.0014, 0.0016, 0.0019, 0.0017, 0.0013, 0.0011, 0.0013,
              0.0012, 0.0039, 0.0015, 0.0019, 0.0019, 0.0023, 0.0025, 0.0017, 0.0029])
y = np.array([10.7, 7.22, 14.8, 12.35, 9.15, 8.42, 11.1, 19.9, 6.46, 10.0,
              12.41, 8.05, 12.3, 8.9, 11.06, 4.2, 17.3, 14.12, 6.37, 18.12,
              26.72, 16.4, 8.91, 15.3, 11.24, 5.61, 11.7, 8.63, 10.4])
x_forecast = 0.021
n = len(x)

print("Лабораторная работа 5. Вариант 12: Cu_волосы = f(Cu_воды)")
print(f"Наблюдений n = {n}, x от {x.min()} до {x.max()}, прогноз нужен при x = {x_forecast}")

print("\n1. РУЧНОЙ РАСЧЁТ ЛИНЕЙНОЙ РЕГРЕССИИ (МНК)")
sum_x, sum_y = x.sum(), y.sum()
sum_xy, sum_x2, sum_y2 = (x * y).sum(), (x ** 2).sum(), (y ** 2).sum()
a1 = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
a0 = (sum_y - a1 * sum_x) / n
r = (n * sum_xy - sum_x * sum_y) / np.sqrt((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2))
print(f"Σx = {sum_x:.4f}   Σy = {sum_y:.2f}   Σxy = {sum_xy:.5f}   Σx² = {sum_x2:.7f}   Σy² = {sum_y2:.4f}")
print(f"a1 = (nΣxy - ΣxΣy) / (nΣx² - (Σx)²) = {a1:.2f}")
print(f"a0 = (Σy - a1·Σx) / n = {a0:.4f}")
print(f"y = {a0:.4f} {a1:+.2f}·x")
print(f"r = {r:.4f}, R² = r² = {r ** 2:.4f}")

print("\n2. МОДЕЛИ: ЛИНЕЙНАЯ, ПОЛИНОМИАЛЬНАЯ, СТЕПЕННАЯ")
c2 = np.polyfit(x, y, 2)
p1, ln_p0 = np.polyfit(np.log(x), np.log(y), 1)
(q0, q1), _ = curve_fit(lambda t, a, b: a * t ** b, x, y, p0=[np.exp(ln_p0), p1], maxfev=20000)
models = {
    "Линейная": (f"y = {a0:.3f} {a1:+.2f}·x", lambda t: a0 + a1 * t, 1),
    "Полиномиальная": (f"y = {c2[2]:.3f} {c2[1]:+.1f}·x {c2[0]:+.0f}·x²", lambda t: np.polyval(c2, t), 2),
    "Степенная": (f"y = {q0:.3f}·x^{q1:.4f}", lambda t: q0 * t ** q1, 1),
}
sst = ((y - y.mean()) ** 2).sum()
r2 = {}
table = {}
print(f"{'модель':15} {'R':>7} {'R²':>8} {'R² скорр.':>10} {'F':>7} {'p':>7} {'ст. ошибка':>11}")
for name, (equation, f, k) in models.items():
    sse = ((y - f(x)) ** 2).sum()
    r2[name] = 1 - sse / sst
    r2_adj = 1 - (1 - r2[name]) * (n - 1) / (n - k - 1)
    F = r2[name] / k / ((1 - r2[name]) / (n - k - 1))
    p_value = stats.f.sf(F, k, n - k - 1)
    std_error = np.sqrt(sse / (n - k - 1))
    table[name] = (np.sqrt(r2[name]), r2[name], r2_adj, F, p_value, std_error)
    print(f"{name:15} {np.sqrt(r2[name]):7.4f} {r2[name]:8.4f} {r2_adj:10.4f} {F:7.3f} {p_value:7.4f} {std_error:11.3f}")
for name, (equation, f, k) in models.items():
    print(f"{name:15} {equation}")
best = max(r2, key=r2.get)
print(f"Лучшая модель: {best}, R² = {r2[best]:.4f} ({r2[best] * 100:.1f} %)")

print("\n3. ТЕСНОТА СВЯЗИ И ЗНАЧИМОСТЬ ЛИНЕЙНОЙ МОДЕЛИ")
if abs(r) < 0.3:
    closeness = "практически отсутствует"
elif abs(r) < 0.5:
    closeness = "слабая"
elif abs(r) < 0.7:
    closeness = "умеренная"
else:
    closeness = "сильная"
direction = "прямая" if r > 0 else "обратная"
print(f"r = {r:.4f}: связь вероятностная, {direction}; теснота связи - {closeness}")
F_linear = r ** 2 / (1 - r ** 2) * (n - 2)
F_critical = stats.f.ppf(0.95, 1, n - 2)
print(f"F = {F_linear:.3f}, критическое F(0,05; 1; {n - 2}) = {F_critical:.3f}")
if F_linear > F_critical:
    print("F больше критического: связь статистически значима")
else:
    print("F меньше критического (p > 0,05): связь статистически НЕ значима")
i_max = y.argmax()
others = np.arange(n) != i_max
r2_without = np.corrcoef(x[others], y[others])[0, 1] ** 2
print(f"Без точки с наибольшим y ({y[i_max]} при x = {x[i_max]}): R² = {r2_without:.4f}")

print("\n4. ОБРАТНАЯ РЕГРЕССИЯ, ДИСПЕРСИИ И ДОВЕРИТЕЛЬНЫЕ ИНТЕРВАЛЫ")
b1 = (n * sum_xy - sum_x * sum_y) / (n * sum_y2 - sum_y ** 2)
b0 = (sum_x - b1 * sum_y) / n
print(f"b1 = (nΣxy - ΣxΣy) / (nΣy² - (Σy)²) = {b1:.7f}")
print(f"b0 = (Σx - b1·Σy) / n = {b0:.5f}")
print(f"Прямая регрессия:   y = {a0:.4f} {a1:+.2f}·x")
print(f"Обратная регрессия: x = {b0:.5f} {b1:+.7f}·y")
print(f"Проверка: a1·b1 = {a1 * b1:.4f} = r²")
t_student = stats.t.ppf(1 - 0.05 / 2, n - 2)
print(f"t(p = 0,05; ν = n - 2 = {n - 2}) = {t_student:.3f}")
intervals = {}
for name, values in (("x", x), ("y", y)):
    mean = values.mean()
    S2 = ((values ** 2).sum() - values.sum() ** 2 / n) / (n - 1)
    S_mean = np.sqrt(S2 / n)
    delta = t_student * S_mean
    digits = 2 if int(f"{delta:e}"[0]) <= 3 else 1
    decimals = digits - 1 - int(np.floor(np.log10(delta)))
    intervals[name] = (mean, S2, S_mean, delta, f"{mean:.{decimals}f} ± {delta:.{decimals}f}")
    print(f"{name}: среднее = {mean:.5g}, S² = {S2:.4g}, S_ср = √(S²/n) = {S_mean:.4g}, Δ = t·S_ср = {delta:.4g}")
    print(f"   результат: {name} = {intervals[name][4]}")

print("\n5. ПРОГНОЗ")
y_best = models[best][1](x_forecast)
print(f"Прогноз возможен при R² >= 75 %. У лучшей модели R² = {r2[best] * 100:.1f} %")
if r2[best] >= 0.75:
    print(f"Прогноз: y({x_forecast}) = {y_best:.3f} мкг/г")
else:
    print("Прогноз невозможен")
    print(f"Формально {best.lower()} модель дала бы y({x_forecast}) = {y_best:.2f}, но ей верить нельзя")
print(f"Значение x = {x_forecast} больше максимума данных {x.max()} в {x_forecast / x.max():.1f} раза: это экстраполяция")

print("\n6. ПОДБОР НЕЛИНЕЙНОЙ РЕГРЕССИИ С R² >= 75 %")
z = 1 / x
h1 = (n * (z * y).sum() - z.sum() * sum_y) / (n * (z ** 2).sum() - z.sum() ** 2)
h0 = (sum_y - h1 * z.sum()) / n
r2_hyper = np.corrcoef(y, h0 + h1 * z)[0, 1] ** 2
print(f"Гипербола (z = 1/x): y = {h0:.3f} {h1:+.5f}/x, R² = {r2_hyper:.4f}")
degrees = [1, 2, 3, 4, 5, 6]
r2_poly = []
for d in degrees:
    c = np.polyfit(x, y, d)
    r2_poly.append(np.corrcoef(y, np.polyval(c, x))[0, 1] ** 2)
    print(f"полином степени {d}: R² = {r2_poly[-1]:.4f}")
if max(r2_poly + [r2_hyper]) < 0.75:
    print("Ни гипербола, ни полиномы до 6-й степени не дают R² >= 75 %")
    print("Рост R² с ростом степени - это подгонка под шум, а не связь")

grid = np.linspace(x.min(), x.max(), 200)
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
for ax, (name, (equation, f, k)) in zip(axes, models.items()):
    ax.scatter(x, y, color="black", s=14)
    ax.plot(grid, f(grid), color="red")
    ax.set_title(name)
    ax.text(0.03, 0.97, f"{equation}\nR² = {r2[name]:.4f}", transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="gray"))
    ax.set_xlabel("Cu_воды, мг/л")
    ax.set_ylabel("Cu_волосы, мкг/г")
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(folder / "01_models.png", dpi=150)
plt.close(fig)

fig = plt.figure(figsize=(7, 4))
plt.bar(degrees, r2_poly)
plt.axhline(0.75, color="red", linestyle="--", label="порог 75 %")
plt.xlabel("степень полинома")
plt.ylabel("R²")
plt.title("R² полиномов разной степени")
plt.legend()
plt.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(folder / "02_poly_degree.png", dpi=150)
plt.close(fig)

y_grid = np.linspace(y.min(), y.max(), 200)
fig = plt.figure(figsize=(7, 4.5))
plt.scatter(x, y, color="black", s=14, label="эксперимент")
plt.plot(grid, a0 + a1 * grid, color="red", label=f"прямая: y = {a0:.3f} {a1:+.2f}·x")
plt.plot(b0 + b1 * y_grid, y_grid, color="blue", label=f"обратная: x = {b0:.5f} {b1:+.7f}·y")
plt.scatter([x.mean()], [y.mean()], color="green", s=50, zorder=3, label="точка средних")
plt.xlabel("Cu_воды, мг/л")
plt.ylabel("Cu_волосы, мкг/г")
plt.title("Прямая и обратная регрессия")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(folder / "03_regression_lines.png", dpi=150)
plt.close(fig)
print(f"\nГрафики сохранены в папке {folder}")

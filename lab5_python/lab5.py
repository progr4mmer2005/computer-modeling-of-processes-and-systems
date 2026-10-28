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
    closeness = "слабая"
elif abs(r) < 0.7:
    closeness = "умеренная"
else:
    closeness = "сильная"
direction = "положительная" if r > 0 else "отрицательная"
print(f"r = {r:.4f}: связь {closeness}, {direction}")
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

print("\n4. ПРОГНОЗ")
y_best = models[best][1](x_forecast)
print(f"Прогноз возможен при R² >= 75 %. У лучшей модели R² = {r2[best] * 100:.1f} %")
if r2[best] >= 0.75:
    print(f"Прогноз: y({x_forecast}) = {y_best:.3f} мкг/г")
else:
    print("Прогноз невозможен")
    print(f"Формально {best.lower()} модель дала бы y({x_forecast}) = {y_best:.2f}, но ей верить нельзя")
print(f"Значение x = {x_forecast} больше максимума данных {x.max()} в {x_forecast / x.max():.1f} раза: это экстраполяция")

print("\n5. ПОЛИНОМЫ РАЗНЫХ СТЕПЕНЕЙ")
degrees = [1, 2, 3, 4, 5, 6]
r2_poly = []
for d in degrees:
    c = np.polyfit(x, y, d)
    r2_poly.append(np.corrcoef(y, np.polyval(c, x))[0, 1] ** 2)
    print(f"степень {d}: R² = {r2_poly[-1]:.4f}")
if max(r2_poly) < 0.75:
    print("Ни один полином до 6-й степени не даёт R² >= 75 %")
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
print(f"\nГрафики сохранены в папке {folder}")

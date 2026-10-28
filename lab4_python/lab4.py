from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

folder = Path(__file__).parent / "results"
folder.mkdir(exist_ok=True)

x = np.array([0.082, 0.088, 0.079, 0.099, 0.101, 0.107, 0.101, 0.115, 0.153, 0.142,
              0.116, 0.117, 0.137, 0.118, 0.128, 0.129, 0.148, 0.152, 0.155, 0.168,
              0.168, 0.17, 0.19, 0.204, 0.207, 0.21, 0.212, 0.222, 0.24])
y = np.array([4.30, 3.33, 4.45, 6.43, 5.87, 3.90, 5.00, 3.22, 6.15, 3.70,
              4.00, 6.04, 7.35, 10.5, 3.55, 6.43, 9.2, 7.22, 5.98, 3.42,
              3.87, 7.41, 4.65, 3.44, 4.66, 8.35, 4.8, 16.7, 7.22])
x_forecast = 0.424
n = len(x)

print("Лабораторная работа 4. Вариант 12: Sr_волосы = f(Sr_воды)")
print(f"Наблюдений n = {n}, x от {x.min()} до {x.max()}, прогноз нужен при x = {x_forecast}")

print("\n1. РУЧНОЙ РАСЧЁТ ЛИНЕЙНОЙ РЕГРЕССИИ (МНК)")
sum_x, sum_y = x.sum(), y.sum()
sum_xy, sum_x2, sum_y2 = (x * y).sum(), (x ** 2).sum(), (y ** 2).sum()
a1 = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
a0 = (sum_y - a1 * sum_x) / n
r = (n * sum_xy - sum_x * sum_y) / np.sqrt((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2))
print(f"Σx = {sum_x:.4f}   Σy = {sum_y:.2f}   Σxy = {sum_xy:.5f}   Σx² = {sum_x2:.6f}   Σy² = {sum_y2:.4f}")
print(f"a1 = (nΣxy - ΣxΣy) / (nΣx² - (Σx)²) = {a1:.4f}")
print(f"a0 = (Σy - a1·Σx) / n = {a0:.4f}")
print(f"y = {a0:.4f} + {a1:.4f}·x")
print(f"r = {r:.4f}, R² = r² = {r ** 2:.4f}")

print("\n2. ПЯТЬ МОДЕЛЕЙ И КОЭФФИЦИЕНТ ДЕТЕРМИНАЦИИ R²")
c2 = np.polyfit(x, y, 2)
l1, l0 = np.polyfit(np.log(x), y, 1)
p1, ln_p0 = np.polyfit(np.log(x), np.log(y), 1)
p0 = np.exp(ln_p0)
e1, ln_e0 = np.polyfit(x, np.log(y), 1)
e0 = np.exp(ln_e0)
models = {
    "Линейная": (f"y = {a0:.3f} {a1:+.3f}·x", lambda t: a0 + a1 * t),
    "Полиномиальная": (f"y = {c2[2]:.3f} {c2[1]:+.3f}·x {c2[0]:+.3f}·x²", lambda t: np.polyval(c2, t)),
    "Логарифмическая": (f"y = {l0:.3f} {l1:+.3f}·ln(x)", lambda t: l0 + l1 * np.log(t)),
    "Степенная": (f"y = {p0:.3f}·x^{p1:.3f}", lambda t: p0 * t ** p1),
    "Экспоненциальная": (f"y = {e0:.3f}·e^({e1:.3f}·x)", lambda t: e0 * np.exp(e1 * t)),
}
r2 = {}
for name, (equation, f) in models.items():
    r2[name] = np.corrcoef(y, f(x))[0, 1] ** 2
    print(f"{name:17} {equation:38} R² = {r2[name]:.4f}")
best = max(r2, key=r2.get)
print(f"Лучшая модель: {best}, R² = {r2[best]:.4f} ({r2[best] * 100:.1f} %)")

print("\n3. ТЕСНОТА СВЯЗИ И СТАТИСТИЧЕСКАЯ ОЦЕНКА ЛИНЕЙНОЙ МОДЕЛИ")
if abs(r) < 0.3:
    closeness = "слабая"
elif abs(r) < 0.7:
    closeness = "умеренная"
else:
    closeness = "сильная"
R2 = r ** 2
R2_adj = 1 - (1 - R2) * (n - 1) / (n - 2)
F = R2 / (1 - R2) * (n - 2)
F_critical = stats.f.ppf(0.95, 1, n - 2)
p_value = stats.f.sf(F, 1, n - 2)
std_error = np.sqrt(((y - (a0 + a1 * x)) ** 2).sum() / (n - 2))
direction = "положительная" if r > 0 else "отрицательная"
print(f"r = {r:.4f}: связь {closeness}, {direction}")
print(f"R² = {R2:.4f}, скорректированный R² = {R2_adj:.4f}, стандартная ошибка = {std_error:.3f}")
print(f"F = {F:.3f}, критическое F(0,05; 1; {n - 2}) = {F_critical:.3f}, p = {p_value:.4f}")
if F > F_critical:
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
print(f"Значение x = {x_forecast} вне диапазона данных [{x.min()}; {x.max()}]: это экстраполяция")

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
fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
for ax, (name, (equation, f)) in zip(axes.flat, models.items()):
    ax.scatter(x, y, color="black", s=14)
    ax.plot(grid, f(grid), color="red")
    ax.set_title(name)
    ax.text(0.03, 0.97, f"{equation}\nR² = {r2[name]:.4f}", transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="gray"))
    ax.set_xlabel("Sr_воды, мг/л")
    ax.set_ylabel("Sr_волосы, мкг/г")
    ax.grid(alpha=0.3)
axes[1, 2].scatter(x, y, color="black", s=14, label="эксперимент")
for name, (equation, f) in models.items():
    axes[1, 2].plot(grid, f(grid), label=name)
axes[1, 2].set_title("Все модели вместе")
axes[1, 2].set_xlabel("Sr_воды, мг/л")
axes[1, 2].legend(fontsize=7)
axes[1, 2].grid(alpha=0.3)
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

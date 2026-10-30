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
    if name in ("Степенная", "Экспоненциальная"):
        r2[name] = np.corrcoef(np.log(y), np.log(f(x)))[0, 1] ** 2
    else:
        r2[name] = np.corrcoef(y, f(x))[0, 1] ** 2
    print(f"{name:17} {equation:38} R² = {r2[name]:.4f}")
print("Для степенной и экспоненциальной моделей R² считается по ln y (как требует методичка)")
best = max(r2, key=r2.get)
print(f"Лучшая модель: {best}, R² = {r2[best]:.4f} ({r2[best] * 100:.1f} %)")

print("\n3. ТЕСНОТА СВЯЗИ И СТАТИСТИЧЕСКАЯ ОЦЕНКА ЛИНЕЙНОЙ МОДЕЛИ")
if abs(r) < 0.3:
    closeness = "практически отсутствует"
elif abs(r) < 0.5:
    closeness = "слабая"
elif abs(r) < 0.7:
    closeness = "умеренная"
else:
    closeness = "сильная"
direction = "прямая" if r > 0 else "обратная"
print(f"r = {r:.4f}: связь вероятностная, {direction}; теснота связи - {closeness} (таблица 4 методички)")
R2 = r ** 2
R2_adj = 1 - (1 - R2) * (n - 1) / (n - 2)
F = R2 / (1 - R2) * (n - 2)
F_critical = stats.f.ppf(0.95, 1, n - 2)
p_value = stats.f.sf(F, 1, n - 2)
std_error = np.sqrt(((y - (a0 + a1 * x)) ** 2).sum() / (n - 2))
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

print("\n4. ОБРАТНАЯ РЕГРЕССИЯ, ДИСПЕРСИИ И ДОВЕРИТЕЛЬНЫЕ ИНТЕРВАЛЫ")
b1 = (n * sum_xy - sum_x * sum_y) / (n * sum_y2 - sum_y ** 2)
b0 = (sum_x - b1 * sum_y) / n
print(f"b1 = (nΣxy - ΣxΣy) / (nΣy² - (Σy)²) = {b1:.5f}")
print(f"b0 = (Σx - b1·Σy) / n = {b0:.4f}")
print(f"Прямая регрессия:   y = {a0:.4f} + {a1:.4f}·x")
print(f"Обратная регрессия: x = {b0:.4f} + {b1:.5f}·y")
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
    print(f"{name}: среднее = {mean:.4f}, S² = {S2:.6g}, S_ср = √(S²/n) = {S_mean:.4g}, Δ = t·S_ср = {delta:.4g}")
    print(f"   результат: {name} = {intervals[name][4]}")

print("\n5. ПРОГНОЗ")
y_best = models[best][1](x_forecast)
print(f"Прогноз возможен при R² >= 75 %. У лучшей модели R² = {r2[best] * 100:.1f} %")
if r2[best] >= 0.75:
    print(f"Прогноз: y({x_forecast}) = {y_best:.3f} мкг/г")
else:
    print("Прогноз невозможен")
    print(f"Формально {best.lower()} модель дала бы y({x_forecast}) = {y_best:.2f}, но ей верить нельзя")
print(f"Значение x = {x_forecast} вне диапазона данных [{x.min()}; {x.max()}]: это экстраполяция")

print("\n6. ПОДБОР НЕЛИНЕЙНОЙ РЕГРЕССИИ С R² >= 75 %")
z = 1 / x
h1 = (n * (z * y).sum() - z.sum() * sum_y) / (n * (z ** 2).sum() - z.sum() ** 2)
h0 = (sum_y - h1 * z.sum()) / n
r2_hyper = np.corrcoef(y, h0 + h1 * z)[0, 1] ** 2
print(f"Гипербола (z = 1/x): y = {h0:.3f} {h1:+.4f}/x, R² = {r2_hyper:.4f}")
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

y_grid = np.linspace(y.min(), y.max(), 200)
fig = plt.figure(figsize=(7, 4.5))
plt.scatter(x, y, color="black", s=14, label="эксперимент")
plt.plot(grid, a0 + a1 * grid, color="red", label=f"прямая: y = {a0:.3f} + {a1:.3f}·x")
plt.plot(b0 + b1 * y_grid, y_grid, color="blue", label=f"обратная: x = {b0:.4f} + {b1:.5f}·y")
plt.scatter([x.mean()], [y.mean()], color="green", s=50, zorder=3, label="точка средних")
plt.xlabel("Sr_воды, мг/л")
plt.ylabel("Sr_волосы, мкг/г")
plt.title("Прямая и обратная регрессия")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(folder / "03_regression_lines.png", dpi=150)
plt.close(fig)
print(f"\nГрафики сохранены в папке {folder}")

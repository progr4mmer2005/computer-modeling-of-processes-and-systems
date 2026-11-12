from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = Path(__file__).parent
folder = here / "results"
folder.mkdir(exist_ok=True)
files = sorted((here.parent / "references").glob("Домохозяйства*.xls*")) + sorted(here.glob("Домохозяйства*.xls*"))

DELTA = 0.0048
HIDDEN = 8
RATE = 0.05
MOMENTUM = 0.9
MAX_EPOCHS = 200000
rng = np.random.default_rng(12)

print("ЛАБОРАТОРНАЯ РАБОТА 6. НЕЙРОСЕТЬ ОБРАТНОГО РАСПРОСТРАНЕНИЯ ОШИБКИ")
print(f"Вариант 12 -> задание 12: обучить сеть с ошибкой Δ < {DELTA}")

print("\n1. ИСХОДНЫЕ ДАННЫЕ")
if files:
    source = files[0]
    if source.suffix.lower() == ".xls":
        import xlrd
        sheet = xlrd.open_workbook(str(source)).sheet_by_index(0)
        table = [sheet.row_values(i) for i in range(sheet.nrows)]
    else:
        import openpyxl
        table = [list(r) for r in openpyxl.load_workbook(source, data_only=True).active.iter_rows(values_only=True)]
    table = [r for r in table if any(c not in (None, "") for c in r)]
    header = [str(c).strip() for c in table[0]]
    rows = [r for r in table[1:] if all(isinstance(c, (int, float)) for c in r[1:len(header)] if c not in (None, ""))]
    names = [str(r[0]).strip() for r in rows]
    values = np.array([[float(c) for c in r[1:len(header)]] for r in rows])
    out_col = next(i for i, h in enumerate(header[1:]) if "доход" in h.lower())
    columns = [h for i, h in enumerate(header[1:]) if i != out_col]
    X = np.delete(values, out_col, axis=1)
    y = values[:, out_col]
    target = header[1:][out_col]
    demo = False
    print(f"Файл: {source.name}")
else:
    demo = True
    names = ["Агрызский", "Азнакаевский", "Аксубаевский", "Актанышский", "Алексеевский", "Алькеевский",
             "Альметьевский", "Апастовский", "Арский", "Атнинский", "Бавлинский", "Балтасинский", "Бугульминский",
             "Буинский", "Верхнеуслонский", "Высокогорский", "Дрожжановский", "Елабужский", "Заинский",
             "Зеленодольский", "Кайбицкий", "Камско-Устьинский", "Кукморский", "Лаишевский", "Лениногорский",
             "Мамадышский", "Менделеевский", "Мензелинский", "Муслюмовский", "Нижнекамский", "Новошешминский",
             "Нурлатский", "Пестречинский", "Рыбно-Слободский", "Сабинский", "Сармановский", "Спасский",
             "Тетюшский", "Тукаевский", "Тюлячинский", "Черемшанский", "Чистопольский", "Ютазинский"]
    n = len(names)
    share = rng.uniform(18, 55, n)
    deficit = 9500 - 60 * share + rng.normal(0, 600, n)
    surplus = 26000 - 280 * share + rng.normal(0, 1500, n)
    mean = 21000 - 190 * share + 0.1 * surplus + rng.normal(0, 900, n)
    variance = (0.45 * surplus + rng.normal(0, 700, n)) ** 2 / 1e6
    y = 0.95 * mean + 0.30 * surplus * (1 - share / 100) - 0.35 * deficit * share / 100 + 160 * np.sqrt(variance) + 3000
    X = np.column_stack([share, deficit, surplus, mean, variance])
    columns = ["Процент домохозяйств с дефицитом, %", "Средний дефицит, руб.", "Средний префицит, руб.",
               "Мат. ожидание распределения, руб.", "Дисперсия распределения, млн руб.²"]
    target = "Средний доход на одно домохозяйство, руб."
    print("ВНИМАНИЕ: файл «Домохозяйства в районах РТ.xls» не найден.")
    print("Используются ДЕМОНСТРАЦИОННЫЕ (выдуманные) данные. Положите файл в папку references и запустите снова.")
n, m = X.shape
print(f"Районов: {n}, входов: {m}, выход: {target}")
for k, c in enumerate(columns, 1):
    print(f"  x{k} = {c}: от {X[:, k - 1].min():.2f} до {X[:, k - 1].max():.2f}")
print(f"  y = {target}: от {y.min():.2f} до {y.max():.2f}")
print("Первые 5 районов:")
for i in range(5):
    print(f"  {names[i]:18}" + "".join(f"{v:12.2f}" for v in X[i]) + f"{y[i]:12.2f}")

print("\n2. НОРМИРОВКА И РАЗБИЕНИЕ")
x_min, x_max, y_min, y_max = X.min(0), X.max(0), y.min(), y.max()
Xn = (X - x_min) / (x_max - x_min)
yn = (y - y_min) / (y_max - y_min)
print("Нормировка в [0; 1]: x' = (x - x_min) / (x_max - x_min), так же для y")
test = np.arange(n) % 5 == 4
train = ~test
print(f"Обучающая выборка: {train.sum()} районов, тестовая (каждый 5-й): {test.sum()} районов")

print("\n3. СТРУКТУРА СЕТИ")
print(f"Многослойный персептрон {m}-{HIDDEN}-1: {m} входов, один скрытый слой из {HIDDEN} нейронов")
print("с сигмоидой f(s) = 1 / (1 + e^(-s)), выходной нейрон линейный")
print(f"Весов и смещений: {m * HIDDEN + HIDDEN + HIDDEN + 1}")
W1 = rng.normal(0, 1, (m, HIDDEN))
b1 = np.zeros(HIDDEN)
W2 = rng.normal(0, 1, (HIDDEN, 1))
b2 = np.zeros(1)
velocity = [np.zeros_like(a) for a in (W1, b1, W2, b2)]

print("\n4. ОБУЧЕНИЕ МЕТОДОМ ОБРАТНОГО РАСПРОСТРАНЕНИЯ ОШИБКИ")
print(f"Скорость обучения η = {RATE}, момент α = {MOMENTUM}")
print(f"Пример распознан, если |y' - ŷ'| < Δ = {DELTA}; обучение до 100 % распознанных или {MAX_EPOCHS} эпох")
A, t = Xn[train], yn[train][:, None]
history = []
for epoch in range(1, MAX_EPOCHS + 1):
    h = 1 / (1 + np.exp(-(A @ W1 + b1)))
    out = h @ W2 + b2
    e = out - t
    worst = np.abs(e).max()
    if epoch == 1 or epoch % 100 == 0:
        history.append((epoch, (e ** 2).mean(), worst))
    if epoch % 20000 == 0:
        print(f"  эпоха {epoch:6}: СКО = {(e ** 2).mean():.2e}, макс |Δ| = {worst:.5f}, "
              f"распознано {np.mean(np.abs(e) < DELTA) * 100:.0f} %")
    if worst < DELTA:
        break
    d_out = 2 * e / len(A)
    d_h = (d_out @ W2.T) * h * (1 - h)
    grads = [A.T @ d_h, d_h.sum(0), h.T @ d_out, d_out.sum(0)]
    for k, (p, g) in enumerate(zip((W1, b1, W2, b2), grads)):
        velocity[k] = MOMENTUM * velocity[k] - RATE * g
        p += velocity[k]
history.append((epoch, (e ** 2).mean(), worst))
print(f"Остановка на эпохе {epoch}: макс |Δ| на обучающей выборке = {worst:.8f} "
      f"({'меньше' if worst < DELTA else 'НЕ меньше'} {DELTA})")


def predict(x):
    xn = (np.atleast_2d(x) - x_min) / (x_max - x_min)
    return ((1 / (1 + np.exp(-(xn @ W1 + b1)))) @ W2 + b2).ravel() * (y_max - y_min) + y_min


print("\n5. РЕЗУЛЬТАТЫ")
y_hat = predict(X)
delta = np.abs(y_hat - y) / (y_max - y_min)
print(f"  {'район':18}{'выборка':>10}{'y факт':>12}{'y сети':>12}{'|Δ| норм.':>12}")
for i in range(n):
    print(f"  {names[i]:18}{'тест' if test[i] else 'обуч.':>10}{y[i]:12.1f}{y_hat[i]:12.1f}{delta[i]:12.5f}")
train_max, train_mean = delta[train].max(), delta[train].mean()
test_max, test_mean = delta[test].max(), delta[test].mean()
rel_test = (np.abs(y_hat - y) / y)[test].mean() * 100
print(f"Обучающая выборка: макс |Δ| = {train_max:.8f}, средняя |Δ| = {train_mean:.5f}, "
      f"распознано {np.mean(delta[train] < DELTA) * 100:.0f} %")
print(f"Тестовая выборка:  макс |Δ| = {test_max:.5f}, средняя |Δ| = {test_mean:.5f}, "
      f"средняя относительная ошибка {rel_test:.2f} %")

print("\n6. СОХРАНЕНИЕ И ИСПОЛЬЗОВАНИЕ СЕТИ")
np.savez(folder / "network.npz", W1=W1, b1=b1, W2=W2, b2=b2, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
saved = np.load(folder / "network.npz")
new = X.mean(0)
xn = (new - saved["x_min"]) / (saved["x_max"] - saved["x_min"])
answer = float(((1 / (1 + np.exp(-(xn @ saved["W1"] + saved["b1"])))) @ saved["W2"] + saved["b2"])[0])
answer = answer * (saved["y_max"] - saved["y_min"]) + saved["y_min"]
print("Сеть сохранена в results/network.npz и загружена обратно")
print("Новый район со средними значениями входов: " + ", ".join(f"x{k + 1} = {v:.2f}" for k, v in enumerate(new)))
print(f"Ответ сохранённой сети: y = {answer:.1f}")

h = np.array(history)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.semilogy(h[:, 0], h[:, 2], label="макс |Δ| (обучающая)")
ax.semilogy(h[:, 0], h[:, 1], label="СКО (обучающая)")
ax.axhline(DELTA, color="red", ls="--", label=f"Δ = {DELTA}")
ax.set_xlabel("эпоха")
ax.set_ylabel("ошибка (нормированная)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(folder / "01_training.png", dpi=130)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y[train], y_hat[train], label="обучающая")
ax.scatter(y[test], y_hat[test], marker="s", label="тестовая")
ax.plot([y.min(), y.max()], [y.min(), y.max()], color="gray", lw=1)
ax.set_xlabel("фактический доход")
ax.set_ylabel("ответ сети")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(folder / "02_fact_vs_net.png", dpi=130)
plt.close(fig)

order = np.argsort(y)
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(range(n), y[order], "o-", label="факт")
ax.plot(range(n), y_hat[order], "x--", label="сеть")
ax.set_xticks(range(n))
ax.set_xticklabels([names[i] for i in order], rotation=90, fontsize=7)
ax.set_ylabel(target)
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(folder / "03_districts.png", dpi=130)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4))
ax.axis("off")
xs_in, xs_h, x_out = 0.1, 0.5, 0.9
pos_in = np.linspace(0.9, 0.1, m)
pos_h = np.linspace(0.95, 0.05, HIDDEN)
for a in pos_in:
    for b in pos_h:
        ax.plot([xs_in, xs_h], [a, b], color="lightgray", lw=0.6)
for b in pos_h:
    ax.plot([xs_h, x_out], [b, 0.5], color="lightgray", lw=0.6)
ax.scatter([xs_in] * m, pos_in, s=300, color="tab:blue", zorder=3)
ax.scatter([xs_h] * HIDDEN, pos_h, s=300, color="tab:orange", zorder=3)
ax.scatter([x_out], [0.5], s=300, color="tab:green", zorder=3)
for k, a in enumerate(pos_in, 1):
    ax.text(xs_in - 0.03, a, f"x{k}", ha="right", va="center")
ax.text(x_out + 0.03, 0.5, "y", va="center")
ax.set_title(f"Персептрон {m}-{HIDDEN}-1")
fig.tight_layout()
fig.savefig(folder / "04_network.png", dpi=130)
plt.close(fig)

print("\nГрафики сохранены в папку results")

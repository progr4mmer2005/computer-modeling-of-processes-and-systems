from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

folder = Path(__file__).parent / "results"
folder.mkdir(exist_ok=True)

DELTA = 0.0048
HIDDEN = 8
RATE = 0.05
MOMENTUM = 0.9
MAX_EPOCHS = 200000
rng = np.random.default_rng(12)

print("ЛАБОРАТОРНАЯ РАБОТА 6. НЕЙРОСЕТЬ ОБРАТНОГО РАСПРОСТРАНЕНИЯ ОШИБКИ")
print(f"Вариант 12 -> задание 12: обучить сеть с ошибкой Δ < {DELTA}")

print("\n1. ИСХОДНЫЕ ДАННЫЕ")
data = [
    ("Агрызский", 27.3, 7786, 17260, 17443, 72.85, 23959),
    ("Азнакаевский", 53.0, 6675, 11834, 10609, 34.84, 14452),
    ("Аксубаевский", 25.0, 8367, 19894, 17438, 70.25, 24651),
    ("Актанышский", 24.6, 7787, 16834, 19483, 58.01, 25862),
    ("Алексеевский", 30.9, 6485, 19095, 16912, 65.50, 23614),
    ("Алькеевский", 26.5, 7700, 17913, 17203, 40.88, 23599),
    ("Альметьевский", 42.8, 7262, 13666, 13089, 24.73, 17486),
    ("Апастовский", 22.3, 7936, 20179, 19311, 111.15, 27120),
    ("Арский", 51.2, 6694, 12799, 13151, 24.77, 16967),
    ("Атнинский", 49.8, 7103, 9920, 12770, 8.38, 15853),
    ("Бавлинский", 18.1, 8087, 22416, 19451, 93.66, 28022),
    ("Балтасинский", 38.0, 7957, 15466, 15595, 64.39, 20915),
    ("Бугульминский", 22.0, 9156, 18697, 19042, 70.26, 26105),
    ("Буинский", 27.5, 8495, 17875, 16941, 69.38, 23493),
    ("Верхнеуслонский", 33.4, 8194, 13362, 16546, 33.92, 21360),
    ("Высокогорский", 34.8, 8071, 15289, 14016, 46.54, 19416),
    ("Дрожжановский", 35.3, 8733, 16383, 14767, 68.21, 20449),
    ("Елабужский", 52.3, 6472, 10287, 11815, 26.90, 15340),
    ("Заинский", 27.6, 7847, 17787, 17442, 63.10, 23948),
    ("Зеленодольский", 25.0, 8365, 18793, 17091, 82.47, 24190),
    ("Кайбицкий", 42.8, 6387, 14086, 14571, 50.28, 19437),
    ("Камско-Устьинский", 53.0, 5387, 12684, 13144, 30.85, 17163),
    ("Кукморский", 52.1, 5842, 11362, 12635, 28.11, 16416),
    ("Лаишевский", 50.6, 6689, 11451, 12739, 28.42, 16469),
    ("Лениногорский", 20.4, 8561, 21101, 19983, 81.50, 27858),
    ("Мамадышский", 52.7, 5419, 14371, 12796, 52.89, 17362),
    ("Менделеевский", 42.0, 5849, 13720, 13360, 36.69, 18187),
    ("Мензелинский", 50.2, 6295, 12700, 12541, 27.89, 16547),
    ("Муслюмовский", 33.1, 7401, 14366, 17310, 28.00, 22317),
    ("Нижнекамский", 26.1, 7903, 20705, 17084, 71.09, 24446),
    ("Новошешминский", 47.3, 7062, 13734, 13435, 52.53, 17922),
    ("Нурлатский", 42.5, 7688, 16105, 14189, 41.59, 19147),
    ("Пестречинский", 46.8, 6830, 12803, 12577, 36.75, 16841),
    ("Рыбно-Слободский", 25.4, 8341, 17310, 16985, 56.74, 23470),
    ("Сабинский", 23.0, 7465, 20897, 18816, 95.29, 26666),
    ("Сармановский", 46.3, 6019, 14179, 13913, 46.27, 18618),
    ("Спасский", 18.7, 8517, 19450, 19661, 67.39, 27173),
    ("Тетюшский", 53.0, 6720, 8954, 10526, 14.63, 13629),
    ("Тукаевский", 23.0, 8556, 20234, 18394, 84.18, 25928),
    ("Тюлячинский", 40.2, 7443, 14486, 15121, 43.54, 19972),
    ("Черемшанский", 33.5, 7960, 20537, 17540, 85.04, 24302),
    ("Чистопольский", 30.0, 8187, 16683, 17577, 53.12, 23509),
    ("Ютазинский", 24.3, 6991, 18403, 18972, 76.08, 26004),
]
columns = ["Процент домохозяйств с дефицитом, %", "Средний дефицит, руб.", "Средний префицит, руб.",
           "Мат. ожидание распределения, руб.", "Дисперсия распределения, млн руб.²"]
target = "Средний доход на одно домохозяйство, руб."
names = [row[0] for row in data]
X = np.array([row[1:6] for row in data])
y = np.array([row[6] for row in data], dtype=float)
n, m = X.shape
print(f"Районов: {n}, входов: {m}, выход: {target}")
for k, c in enumerate(columns, 1):
    print(f"  x{k} = {c}: от {X[:, k - 1].min():.2f} до {X[:, k - 1].max():.2f}")
print(f"  y = {target}: от {y.min():.2f} до {y.max():.2f}")
print(f"  {'№':>3} {'район':18}{'x1':>7}{'x2':>8}{'x3':>8}{'x4':>8}{'x5':>9}{'y':>8}")
for i in range(n):
    print(f"  {i + 1:3} {names[i]:18}{X[i, 0]:7.1f}{X[i, 1]:8.0f}{X[i, 2]:8.0f}{X[i, 3]:8.0f}{X[i, 4]:9.2f}{y[i]:8.0f}")

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

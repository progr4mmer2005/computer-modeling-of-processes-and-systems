from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

folder = Path(__file__).parent / "results"
folder.mkdir(exist_ok=True)

ROWS, COLS = 3, 3
EPOCHS = 500
TARGET = "Республика Татарстан"
COMPARE = ["Республика Башкортостан", "Удмуртская Республика", "Чувашская Республика", "Республика Марий Эл",
           "Республика Мордовия", "Пермский край", "Кировская область", "Нижегородская область",
           "Оренбургская область", "Пензенская область", "Самарская область", "Саратовская область",
           "Ульяновская область"]
rng = np.random.default_rng(12)

print("ЛАБОРАТОРНАЯ РАБОТА 7. НЕЙРОННАЯ СЕТЬ ДЛЯ КЛАСТЕРИЗАЦИИ (КАРТА КОХОНЕНА)")
print("Вариант 12 -> задание 2")

print("\n1. ИСХОДНЫЕ ДАННЫЕ")
data = [
    ("Республика Татарстан", 474.6, 71.3, 261.3, 59.6),
    ("Республика Башкортостан", 438.4, 35.1, 135.3, 24.1),
    ("Удмуртская Республика", 119.6, 14.3, 44.7, 8.6),
    ("Чувашская Республика", 63.4, 7.6, 22.7, 5.0),
    ("Республика Марий Эл", 48.1, 4.3, 12.8, 2.1),
    ("Республика Мордовия", 57.6, 4.8, 15.9, 2.7),
    ("Пермский край", 231.9, 27.7, 105.5, 20.6),
    ("Кировская область", 93.7, 6.9, 31.6, 4.5),
    ("Нижегородская область", 313.2, 34.9, 149.3, 31.2),
    ("Оренбургская область", 157.0, 15.1, 57.4, 12.8),
    ("Пензенская область", 97.3, 8.4, 34.7, 5.1),
    ("Самарская область", 375.4, 47.4, 157.1, 28.4),
    ("Саратовская область", 174.9, 19.7, 61.4, 11.8),
    ("Ульяновская область", 102.6, 8.7, 37.2, 6.8),
    ("г. Москва", 5622.9, 1511.4, 7192.4, 3797.3),
    ("Московская область", 1486.3, 239.5, 1052.9, 249.9),
    ("Воронежская область", 181.3, 22.8, 96.2, 19.5),
    ("Белгородская область", 124.4, 14.2, 63.1, 13.2),
    ("Тульская область", 134.6, 16.5, 63.4, 10.9),
    ("Ярославская область", 122.7, 11.6, 41.7, 9.9),
    ("г. Санкт-Петербург", 1267.6, 237.3, 994.3, 326.6),
    ("Ленинградская область", 209.4, 17.8, 73.3, 16.9),
    ("Мурманская область", 100.0, 11.0, 59.1, 11.9),
    ("Калининградская область", 101.6, 13.2, 52.0, 8.5),
    ("Краснодарский край", 682.5, 75.0, 251.1, 56.4),
    ("Ростовская область", 291.6, 39.4, 156.6, 27.3),
    ("Волгоградская область", 182.7, 18.9, 67.3, 13.2),
    ("Республика Крым", 90.0, 6.7, 23.3, 3.7),
    ("г. Севастополь", 27.4, 2.6, 6.6, 1.4),
    ("Республика Дагестан", 98.7, 6.0, 14.6, 1.3),
    ("Ставропольский край", 221.9, 21.3, 60.1, 9.4),
    ("Чеченская Республика", 37.6, 1.8, 6.8, 0.4),
    ("Свердловская область", 479.2, 66.7, 210.5, 52.8),
    ("Челябинская область", 393.5, 36.8, 126.4, 23.3),
    ("Тюменская область", 260.7, 40.4, 153.8, 37.7),
    ("Ханты-Мансийский АО", 335.1, 59.3, 218.2, 77.3),
    ("Новосибирская область", 237.3, 33.0, 138.9, 30.9),
    ("Красноярский край", 277.6, 42.9, 176.5, 36.6),
    ("Кемеровская область", 228.4, 26.7, 90.3, 14.1),
    ("Иркутская область", 222.5, 30.8, 83.8, 19.3),
    ("Омская область", 146.4, 14.5, 50.2, 10.4),
    ("Приморский край", 206.2, 25.3, 75.9, 18.5),
    ("Хабаровский край", 160.9, 22.9, 84.2, 19.6),
    ("Республика Саха (Якутия)", 151.7, 20.8, 66.5, 18.1),
]
columns = ["Вклады физлиц в рублях, млрд руб.", "Вклады физлиц в валюте и драгметаллах, млрд руб.",
           "Средства организаций в рублях, млрд руб.", "Средства организаций в валюте и драгметаллах, млрд руб."]
names = [row[0] for row in data]
X = np.array([row[1:] for row in data])
n, m = X.shape
print(f"Объектов (регионов): {n}, признаков: {m}")
for k, c in enumerate(columns, 1):
    print(f"  x{k} = {c}: от {X[:, k - 1].min():.1f} до {X[:, k - 1].max():.1f}")
print(f"Анализируемый объект: {TARGET}; сравнение с регионами Приволжского ФО ({len(COMPARE)})")

print("\n2. ПОДГОТОВКА ДАННЫХ")
L = np.log10(X)
low, high = L.min(0), L.max(0)
Z = (L - low) / (high - low)
print("Суммы различаются в сотни раз (Москва), поэтому берётся lg(x), затем нормировка в [0; 1]:")
print("  z = (lg x - lg x_min) / (lg x_max - lg x_min)")

print("\n3. КАРТА КОХОНЕНА")
grid = np.array([(r, c) for r in range(ROWS) for c in range(COLS)], dtype=float)
W = rng.uniform(0.2, 0.8, (ROWS * COLS, m))
print(f"Карта {ROWS}×{COLS} = {ROWS * COLS} нейронов (кластеров), вектор весов каждого нейрона имеет длину {m} (по числу признаков)")
print("Обучение без учителя:")
print("  1) нейрон-победитель j* = argmin ||z - w_j|| (ближайший по евклидову расстоянию)")
print("  2) w_j = w_j + η(t)·h(j, j*)·(z - w_j), h = exp(-d²(j, j*) / (2σ(t)²))")
print(f"  η убывает от 0,5 до 0,01, σ убывает от 1,5 до 0,3, эпох: {EPOCHS}")
errors = []
for epoch in range(EPOCHS):
    share = epoch / (EPOCHS - 1)
    eta = 0.5 * (0.01 / 0.5) ** share
    sigma = 1.5 * (0.3 / 1.5) ** share
    for i in rng.permutation(n):
        win = np.argmin(((W - Z[i]) ** 2).sum(1))
        d2 = ((grid - grid[win]) ** 2).sum(1)
        W += eta * np.exp(-d2 / (2 * sigma ** 2))[:, None] * (Z[i] - W)
    dist = np.sqrt(((Z[:, None, :] - W[None, :, :]) ** 2).sum(2))
    errors.append(dist.min(1).mean())
    if epoch in (0, 99, 299, EPOCHS - 1):
        print(f"  эпоха {epoch + 1:4}: η = {eta:.3f}, σ = {sigma:.2f}, средняя ошибка квантования = {errors[-1]:.4f}")
cluster = dist.argmin(1)

print("\n4. КЛАСТЕРЫ")
print("Номер кластера = (строка, столбец) нейрона на карте")
centers = {}
for j in range(ROWS * COLS):
    members = [names[i] for i in range(n) if cluster[i] == j]
    label = f"{int(grid[j, 0]) + 1}.{int(grid[j, 1]) + 1}"
    if not members:
        print(f"  кластер {label}: пусто")
        continue
    centers[j] = X[cluster == j].mean(0)
    print(f"  кластер {label} ({len(members)}): " + ", ".join(members))
print("Средние значения признаков по кластерам:")
print(f"  {'кластер':8}" + "".join(f"{'x' + str(k + 1):>10}" for k in range(m)))
for j, c in centers.items():
    print(f"  {int(grid[j, 0]) + 1}.{int(grid[j, 1]) + 1:<6}" + "".join(f"{v:10.1f}" for v in c))

print("\n5. ПОЛОЖЕНИЕ АНАЛИЗИРУЕМОГО ОБЪЕКТА")
t = names.index(TARGET)
tj = cluster[t]
t_label = f"{int(grid[tj, 0]) + 1}.{int(grid[tj, 1]) + 1}"
print(f"{TARGET}: кластер {t_label}")
print("  признак" + " " * 50 + "значение   среднее по всем   место из " + str(n))
for k, c in enumerate(columns):
    rank = int((X[:, k] > X[t, k]).sum()) + 1
    print(f"  {c:56}{X[t, k]:10.1f}{X[:, k].mean():16.1f}{rank:12}")
same = [names[i] for i in range(n) if cluster[i] == tj and i != t]
print("В том же кластере: " + (", ".join(same) if same else "нет"))

print("\n6. СРАВНЕНИЕ С РЕГИОНАМИ ПРИВОЛЖСКОГО ФО")
compare = [names.index(c) for c in COMPARE if c in names]
rows_cmp = []
for i in compare:
    d = np.sqrt(((Z[i] - Z[t]) ** 2).sum())
    j = cluster[i]
    rows_cmp.append((d, names[i], f"{int(grid[j, 0]) + 1}.{int(grid[j, 1]) + 1}", X[i].sum()))
rows_cmp.sort()
print(f"  {'регион':28}{'кластер':>9}{'расстояние до РТ':>18}{'всего, млрд':>14}")
print(f"  {TARGET:28}{t_label:>9}{0:18.3f}{X[t].sum():14.1f}")
for d, name, label, total in rows_cmp:
    print(f"  {name:28}{label:>9}{d:18.3f}{total:14.1f}")
pfo_rank = sorted([X[t].sum()] + [r[3] for r in rows_cmp], reverse=True).index(X[t].sum()) + 1
print(f"По сумме средств {TARGET} на {pfo_rank}-м месте из {len(rows_cmp) + 1} регионов ПФО")
print(f"Ближайший по структуре: {rows_cmp[0][1]}, самый далёкий: {rows_cmp[-1][1]}")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, EPOCHS + 1), errors)
ax.set_xlabel("эпоха")
ax.set_ylabel("средняя ошибка квантования")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(folder / "01_training.png", dpi=130)
plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for k, ax in enumerate(axes.flat):
    values = 10 ** (W[:, k] * (high[k] - low[k]) + low[k])
    im = ax.imshow(values.reshape(ROWS, COLS), cmap="YlOrRd")
    for j in range(ROWS * COLS):
        ax.text(grid[j, 1], grid[j, 0], f"{values[j]:.0f}", ha="center", va="center", fontsize=9)
    ax.set_title(columns[k], fontsize=9)
    ax.set_xticks(range(COLS))
    ax.set_yticks(range(ROWS))
    ax.set_xticklabels(range(1, COLS + 1))
    ax.set_yticklabels(range(1, ROWS + 1))
    fig.colorbar(im, ax=ax, shrink=0.8)
fig.suptitle("Раскраска карты по признакам (веса нейронов, млрд руб.)")
fig.tight_layout()
fig.savefig(folder / "02_feature_maps.png", dpi=130)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 7))
counts = np.bincount(cluster, minlength=ROWS * COLS).reshape(ROWS, COLS)
ax.imshow(counts, cmap="Blues", alpha=0.6)
for j in range(ROWS * COLS):
    members = [names[i] for i in range(n) if cluster[i] == j]
    text = "\n".join(("► " if s == TARGET else "") + s.replace("Республика", "Респ.").replace("область", "обл.")
                     for s in members)
    ax.text(grid[j, 1], grid[j, 0], f"кластер {int(grid[j, 0]) + 1}.{int(grid[j, 1]) + 1}\n{text}",
            ha="center", va="center", fontsize=8)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(f"Карта Кохонена {ROWS}×{COLS}: регионы по кластерам (► {TARGET})")
fig.tight_layout()
fig.savefig(folder / "03_clusters.png", dpi=130)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 5))
cmp_names = [TARGET] + [r[1] for r in rows_cmp]
cmp_idx = [names.index(s) for s in cmp_names]
bottom = np.zeros(len(cmp_idx))
for k in range(m):
    ax.bar(range(len(cmp_idx)), X[cmp_idx, k], bottom=bottom, label=columns[k])
    bottom += X[cmp_idx, k]
ax.set_xticks(range(len(cmp_idx)))
ax.set_xticklabels([s.replace("Республика", "Респ.").replace("область", "обл.") for s in cmp_names],
                   rotation=60, ha="right", fontsize=8)
ax.set_ylabel("млрд руб.")
ax.legend(fontsize=7)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(folder / "04_compare.png", dpi=130)
plt.close(fig)

print("\nГрафики сохранены в папку results")

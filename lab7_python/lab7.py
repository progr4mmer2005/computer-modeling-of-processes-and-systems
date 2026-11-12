from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = Path(__file__).parent
folder = here / "results"
folder.mkdir(exist_ok=True)
files = sorted((here.parent / "references").glob("Банковские*.xls*")) + sorted(here.glob("Банковские*.xls*"))

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
    rows = [r for r in table[1:] if all(isinstance(c, (int, float)) for c in r[1:len(header)])]
    names = [str(r[0]).strip() for r in rows]
    X = np.array([[float(c) for c in r[1:len(header)]] for r in rows])
    columns = header[1:]
    demo = False
    print(f"Файл: {source.name}")
else:
    demo = True
    regions = {
        "Республика Татарстан": (4.0, 1.25), "Республика Башкортостан": (4.0, 0.95), "Удмуртская Республика": (1.5, 0.85),
        "Чувашская Республика": (1.2, 0.7), "Республика Марий Эл": (0.7, 0.65), "Республика Мордовия": (0.8, 0.7),
        "Пермский край": (2.5, 1.05), "Кировская область": (1.2, 0.75), "Нижегородская область": (3.2, 1.1),
        "Оренбургская область": (1.9, 0.85), "Пензенская область": (1.3, 0.75), "Самарская область": (3.2, 1.15),
        "Саратовская область": (2.4, 0.8), "Ульяновская область": (1.2, 0.8),
        "г. Москва": (13.0, 4.0), "Московская область": (8.5, 1.8), "Воронежская область": (2.3, 1.0),
        "Белгородская область": (1.5, 1.05), "Тульская область": (1.5, 0.95), "Ярославская область": (1.2, 1.0),
        "г. Санкт-Петербург": (5.6, 2.2), "Ленинградская область": (2.0, 1.0), "Мурманская область": (0.7, 1.4),
        "Калининградская область": (1.0, 1.1), "Краснодарский край": (5.8, 1.1), "Ростовская область": (4.2, 0.95),
        "Волгоградская область": (2.5, 0.8), "Республика Крым": (1.9, 0.5), "г. Севастополь": (0.5, 0.6),
        "Республика Дагестан": (3.2, 0.3), "Ставропольский край": (2.8, 0.75), "Чеченская Республика": (1.5, 0.25),
        "Свердловская область": (4.3, 1.25), "Челябинская область": (3.4, 1.0), "Тюменская область": (1.6, 1.6),
        "Ханты-Мансийский АО": (1.7, 2.0), "Новосибирская область": (2.8, 1.15), "Красноярский край": (2.8, 1.2),
        "Кемеровская область": (2.6, 0.9), "Иркутская область": (2.4, 1.0), "Омская область": (1.9, 0.85),
        "Приморский край": (1.9, 1.1), "Хабаровский край": (1.3, 1.25), "Республика Саха (Якутия)": (1.0, 1.4),
    }
    names = list(regions)
    size = np.array([v[0] for v in regions.values()])
    wealth = np.array([v[1] for v in regions.values()])
    noise = rng.lognormal(0, 0.12, (len(names), 4))
    X = np.column_stack([
        95 * size * wealth,
        11 * size * wealth ** 1.6,
        40 * size * wealth ** 1.8,
        8 * size * wealth ** 2.4,
    ]) * noise
    columns = ["Вклады физлиц в рублях, млрд руб.", "Вклады физлиц в валюте и драгметаллах, млрд руб.",
               "Средства организаций в рублях, млрд руб.", "Средства организаций в валюте и драгметаллах, млрд руб."]
    print("ВНИМАНИЕ: файл «Банковские депозиты (вклады).xls» не найден.")
    print("Используются ДЕМОНСТРАЦИОННЫЕ (выдуманные) данные. Положите файл в папку references и запустите снова.")
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

import contextlib
import io
import runpy

from report_kit import ROOT, Report, console_png, num

LAB = ROOT / "lab7_python"
RES = LAB / "results"

buffer = io.StringIO()
with contextlib.redirect_stdout(buffer):
    v = runpy.run_path(str(LAB / "lab7.py"), run_name="__main__")
lines = buffer.getvalue().splitlines()
OUT = ROOT / "Отчет_лабораторная_7_вариант_12.docx"
X, names, columns, n, m = v["X"], v["names"], v["columns"], v["n"], v["m"]
R, C, grid, cluster = v["ROWS"], v["COLS"], v["grid"], v["cluster"]
TARGET, t = v["TARGET"], v["t"]


def big(x):
    return f"{x:,.0f}".replace(",", " ")


def label(j):
    return f"{int(grid[j, 0]) + 1}.{int(grid[j, 1]) + 1}"


rep = Report(7, "Нейронная сеть для кластеризации")

rep.h1("Цель работы")
rep.p("Изучить самоорганизующуюся карту Кохонена и выполнить с её помощью кластеризацию объектов.")

rep.h1("Задание")
rep.p("Вариант 12, тема 2, задание № 2: спроектировать, обучить и сохранить нейронную сеть Кохонена для "
      "кластеризации средств организаций, банковских депозитов (вкладов) и других привлечённых средств "
      "юридических и физических лиц в рублях, иностранной валюте и драгоценных металлах. Оценить положение "
      f"анализируемого объекта ({TARGET}) и сравнить его с другими объектами (регионы того же федерального округа).")
rep.p("Средства организаций – сумма средств на счетах организаций и депозитов юридических лиц, млн руб.")
rep.table(["Обозначение", "Признак"], [[f"x_{{{k}}}", c] for k, c in enumerate(columns, 1)],
          "Признаки объектов", widths=[3.0, 11.0], align=["center", "left"])
rep.table(["Регион"] + [f"x_{{{k}}}" for k in range(1, m + 1)],
          [[names[i]] + [big(x) for x in X[i]] for i in range(n)],
          f"Исходные данные, {n} регионов", widths=[5.5] + [2.4] * m, size=8, align=["left"] + ["center"] * m)

rep.h1("Решение")
rep.h2("1. Подготовка данных")
lo, hi = X.min(0), X.max(0)
rep.p("Суммы по регионам различаются в тысячи раз, у части регионов есть нулевые значения, поэтому признаки "
      "логарифмируются со сдвигом на 1 и нормируются в [0; 1]:")
rep.formula("z = (lg(x + 1) − lg(x_{min} + 1)) / (lg(x_{max} + 1) − lg(x_{min} + 1)).")
zt = v["Z"][t, 0]
rep.p(f"Для {TARGET}, x_{{1}} = {big(X[t, 0])}: z = (lg {big(X[t, 0] + 1)} − lg {big(lo[0] + 1)}) / "
      f"(lg {big(hi[0] + 1)} − lg {big(lo[0] + 1)}) = {num(zt, 4)}.")

rep.h2("2. Карта Кохонена и обучение")
rep.p(f"Карта {R}×{C} = {R * C} нейронов, каждый нейрон – кластер с вектором весов длины {m}. Обучение без "
      "учителя: для каждого объекта находится нейрон-победитель, затем веса победителя и его соседей "
      "сдвигаются к объекту:")
rep.formula("j* = argmin_{j} ||z − w_{j}||,   ||z − w_{j}|| = √(Σ_{k} (z_{k} − w_{jk})^{2}),")
rep.formula("w_{j} ← w_{j} + η(t)·h(j, j*)·(z − w_{j}),   h(j, j*) = exp(−d^{2}(j, j*) / (2σ(t)^{2})),")
rep.p(f"где d(j, j*) – расстояние между нейронами на сетке карты, η – скорость обучения (убывает от 0,5 до "
      f"0,01), σ – радиус соседства (убывает от 1,5 до 0,3), эпох {v['EPOCHS']}. Качество – средняя ошибка "
      f"квантования (расстояние от объекта до его нейрона); в конце обучения {num(v['errors'][-1], 4)}.")

rep.h2("3. Кластеры")
rows = []
for j in range(R * C):
    members = [names[i] for i in range(n) if cluster[i] == j]
    if members:
        rows.append([label(j), len(members), ", ".join(members)] + [big(x) for x in v["centers"][j]])
rep.table(["Кластер", "Число", "Регионы"] + [f"x_{{{k}}}" for k in range(1, m + 1)], rows,
          "Состав кластеров и средние значения признаков, млн руб.",
          widths=[1.6, 1.3, 7.0] + [1.65] * m, size=7.5, align=["center", "center", "left"] + ["center"] * m)
rep.figure(RES / "03_clusters.png", f"Карта Кохонена {R}×{C} с регионами", 15.0)
rep.figure(RES / "02_feature_maps.png", "Раскраска карты по признакам", 15.0)

rep.h2("4. Положение анализируемого объекта")
tj = cluster[t]
same = [names[i] for i in range(n) if cluster[i] == tj and i != t]
ranks = [int((X[:, k] > X[t, k]).sum()) + 1 for k in range(m)]
rep.p(f"{TARGET} отнесена к кластеру {label(tj)}" + (f" вместе с регионами: {', '.join(same)}." if same else "."))
rep.table(["Признак", "Значение", "Среднее", f"Место из {n}"],
          [[columns[k], big(X[t, k]), big(X[:, k].mean()), ranks[k]] for k in range(m)],
          f"Показатели: {TARGET}", widths=[8.0, 2.2, 2.2, 2.2], align=["left", "center", "center", "center"])

rep.h2(f"5. Сравнение с регионами: {v['DISTRICT']}")
rows_cmp = v["rows_cmp"]
rep.table(["Регион", "Кластер", "Расстояние до объекта", "Всего, млн руб."],
          [[TARGET, label(tj), "0", big(X[t].sum())]] + [[s, lab, num(d, 3), big(tot)]
                                                            for d, s, lab, tot in rows_cmp],
          f"Сравнение с регионами округа: {v['DISTRICT']} (расстояние в нормированных признаках)", widths=[6.0, 2.0, 3.5, 3.0],
          size=8, align=["left", "center", "center", "center"])

rep.h2("6. Вывод программы")
s4 = next(i for i, s in enumerate(lines) if s.startswith("4. КЛАСТЕРЫ"))
s5 = next(i for i, s in enumerate(lines) if s.startswith("5. ПОЛОЖЕНИЕ"))
shots = [(lines[:s4 - 1], "данные и обучение"), (lines[s4:s5 - 1], "кластеры"), (lines[s5:], "анализ объекта")]
for k, (chunk, caption) in enumerate(shots, 1):
    png = RES / f"console_{k}.png"
    console_png(chunk, png, title="run_lab7.bat", font_size=26, light=True)
    rep.figure(png, "Вывод программы lab7.py: " + caption, 16.0)

rep.h1("Выводы")
counts = [int((cluster == j).sum()) for j in range(R * C)]
rep.p(f"1. Карта Кохонена {R}×{C} разбила {n} регионов на {sum(1 for c in counts if c)} кластеров; в одном "
      "кластере оказались регионы с близкими объёмами и структурой средств.")
rep.p(f"2. {TARGET} входит в кластер {label(tj)}, по признакам занимает места {', '.join(map(str, ranks))} "
      f"из {n}.")
rep.p(f"3. Среди регионов округа ({v['DISTRICT']}) {TARGET} на {v['pfo_rank']}-м месте по сумме средств; ближе всего по "
      f"структуре {rows_cmp[0][1]}, дальше всего {rows_cmp[-1][1]}.")

rep.structural("Приложение А")
rep.p("Листинг программы", align="center", indent=False)
rep.code_columns([("lab7.py", (LAB / "lab7.py").read_text(encoding="utf-8")),
                  ("run_lab7.bat", (LAB / "run_lab7.bat").read_text(encoding="utf-8"))])

rep.save(OUT)
print("saved", OUT)

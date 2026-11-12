import contextlib
import io
import runpy

from report_kit import ROOT, Report, console_png, num

LAB = ROOT / "lab6_python"
RES = LAB / "results"

buffer = io.StringIO()
with contextlib.redirect_stdout(buffer):
    v = runpy.run_path(str(LAB / "lab6.py"), run_name="__main__")
lines = buffer.getvalue().splitlines()
demo = v["demo"]
OUT = ROOT / ("Отчет_лабораторная_6_вариант_12" + ("_ДЕМО_ДАННЫЕ" if demo else "") + ".docx")
X, y, names, columns, target = v["X"], v["y"], v["names"], v["columns"], v["target"]
n, m, H = v["n"], v["m"], v["HIDDEN"]
train, test, delta = v["train"], v["test"], v["delta"]

rep = Report(6, "Нейросеть обратного распространения ошибки")

rep.h1("Цель работы")
rep.p("Изучить многослойный персептрон и метод обратного распространения ошибки, научиться проектировать, "
      "обучать и сохранять нейронную сеть. Расчёты выполнены на языке Python (NumPy, Matplotlib), он заменяет "
      "пакеты Excel Neural Package, Deductor и Statistica.")

rep.h1("Задание")
rep.p(f"Вариант 12, тема 1, задание № 12: спроектировать, обучить и сохранить нейронную сеть с ошибкой "
      f"Δ < {num(v['DELTA'], 4)} (Δ – разница между реальными и вычисленными значениями) для расчёта среднего "
      "дохода на одно домохозяйство в районах Республики Татарстан в зависимости от процента домохозяйств с "
      "дефицитом, среднего дефицита, среднего префицита, математического ожидания и дисперсии распределения.")
if demo:
    rep.p("**Файл «Домохозяйства в районах РТ.xls» отсутствует, расчёт выполнен на демонстрационных данных.**")
rep.table(["Обозначение", "Показатель"], [[f"x_{{{k}}}", c] for k, c in enumerate(columns, 1)] + [["y", target]],
          "Входы и выход сети", widths=[3.0, 11.0], align=["center", "left"])
rows = []
half = (n + 1) // 2
for i in range(half):
    row = []
    for k in (i, i + half):
        row += [names[k], num(y[k], 0)] if k < n else ["", ""]
    rows.append(row)
rep.p(f"Районов n = {n}. Значения выхода приведены в таблице 2, все исходные данные – в выводе программы.")
rep.table(["Район", "y", "Район", "y"], rows, "Средний доход по районам", widths=[4.5, 2.5, 4.5, 2.5], size=8,
          align=["left", "center", "left", "center"])

rep.h1("Решение")
rep.h2("1. Нормировка и выборки")
rep.p("Входы и выход приводятся к отрезку [0; 1]:")
rep.formula("x′ = (x − x_{min}) / (x_{max} − x_{min}),   y′ = (y − y_{min}) / (y_{max} − y_{min}).")
rep.p(f"Например, y_{{min}} = {num(y.min(), 1)}, y_{{max}} = {num(y.max(), 1)}; для y = {num(y[0], 1)}: "
      f"y′ = ({num(y[0], 1)} − {num(y.min(), 1)}) / ({num(y.max(), 1)} − {num(y.min(), 1)}) = "
      f"{num((y[0] - y.min()) / (y.max() - y.min()), 4)}. Каждый пятый район ({test.sum()}) отнесён к тестовой "
      f"выборке, остальные {train.sum()} – к обучающей.")

rep.h2("2. Структура сети")
rep.p(f"Многослойный персептрон {m}-{H}-1: {m} входов, один скрытый слой из {H} нейронов с сигмоидой, "
      f"линейный выходной нейрон, {m * H + H + H + 1} весов и смещений:")
rep.formula("h_{j} = f(Σ_{i} w_{ij}·x′_{i} + b_{j}),   f(s) = 1 / (1 + e^{−s}),   ŷ′ = Σ_{j} v_{j}·h_{j} + c,")
rep.p("где w_{ij}, v_{j} – веса, b_{j}, c – смещения, h_{j} – выход j-го скрытого нейрона.")
rep.figure(RES / "04_network.png", f"Структура персептрона {m}-{H}-1", 11.0)

rep.h2("3. Обучение методом обратного распространения ошибки")
rep.p("Ошибка на обучающей выборке и поправки весов (градиентный спуск с моментом):")
rep.formula("E = (1/N)·Σ (ŷ′ − y′)^{2},")
rep.formula("δ = 2(ŷ′ − y′)/N,   δ_{j} = δ·v_{j}·h_{j}(1 − h_{j}),")
rep.formula("Δw(t) = α·Δw(t − 1) − η·∂E/∂w,   ∂E/∂v_{j} = Σ δ·h_{j},   ∂E/∂w_{ij} = Σ δ_{j}·x′_{i},")
rep.p(f"где η = {num(v['RATE'], 2)} – скорость обучения, α = {num(v['MOMENTUM'], 1)} – момент. Пример считается "
      f"распознанным, если |ŷ′ − y′| < Δ = {num(v['DELTA'], 4)}; обучение идёт до распознавания всех примеров.")
reached = v["worst"] < v["DELTA"]
rep.p(f"Обучение остановлено на эпохе {v['epoch']}: наибольшая ошибка на обучающей выборке "
      f"{num(v['worst'], 8)} {'<' if reached else '≥'} {num(v['DELTA'], 4)} (рисунок 2).")
rep.figure(RES / "01_training.png", "Изменение ошибки при обучении", 13.0)

rep.h2("4. Результаты")
rows = [[names[i], "тест" if test[i] else "обуч.", num(y[i], 1), num(v["y_hat"][i], 1), num(delta[i], 5)]
        for i in range(n)]
rep.table(["Район", "Выборка", "y факт", "y сети", "|Δ|"], rows, "Ответы сети", widths=[4.5, 2.0, 2.6, 2.6, 2.2],
          size=8, align=["left", "center", "center", "center", "center"])
rep.p(f"Обучающая выборка: наибольшая |Δ| = {num(v['train_max'], 8)}, средняя {num(v['train_mean'], 5)}. "
      f"Тестовая выборка: наибольшая |Δ| = {num(v['test_max'], 5)}, средняя {num(v['test_mean'], 5)}, средняя "
      f"относительная ошибка {num(v['rel_test'], 2)} %.")
rep.figure(RES / "02_fact_vs_net.png", "Фактический доход и ответ сети", 9.5)
rep.figure(RES / "03_districts.png", "Доход по районам: факт и сеть", 16.0)

rep.h2("5. Сохранение сети")
rep.p(f"Веса, смещения и границы нормировки сохранены в файл results/network.npz. Сеть загружена из файла и "
      f"применена к новому району со средними значениями входов: y = {num(v['answer'], 1)}.")

rep.h2("6. Вывод программы")
s4 = next(i for i, s in enumerate(lines) if s.startswith("4. ОБУЧЕНИЕ"))
s5 = next(i for i, s in enumerate(lines) if s.startswith("5. РЕЗУЛЬТАТЫ"))
s6 = next(i for i, s in enumerate(lines) if s.startswith("6. СОХРАНЕНИЕ"))
summary = [s for s in lines[s5:s6] if "выборка:" in s]
shots = [(lines[:s4 - 1], "данные, нормировка, структура"),
         (lines[s4:s5 - 1] + [""] + summary + [""] + lines[s6:], "обучение и сохранение")]
for k, (chunk, caption) in enumerate(shots, 1):
    png = RES / f"console_{k}.png"
    console_png(chunk, png, title="run_lab6.bat", font_size=26, light=True)
    rep.figure(png, "Вывод программы lab6.py: " + caption, 16.0)

rep.h1("Выводы")
rep.p(f"1. Спроектирован персептрон {m}-{H}-1 для расчёта среднего дохода домохозяйства по {m} показателям.")
rep.p(f"2. Сеть обучена методом обратного распространения ошибки за {v['epoch']} эпох; наибольшая ошибка на "
      f"обучающей выборке {num(v['worst'], 8)} {'меньше' if reached else 'не меньше'} заданной Δ = "
      f"{num(v['DELTA'], 4)}.")
rep.p(f"3. На тестовых районах, не участвовавших в обучении, средняя относительная ошибка {num(v['rel_test'], 2)} %, "
      "сеть пригодна для расчёта дохода по новым данным.")
rep.p("4. Обученная сеть сохранена в файл и использована повторно без обучающих данных.")

rep.structural("Приложение А")
rep.p("Листинг программы", align="center", indent=False)
rep.code_columns([("lab6.py", (LAB / "lab6.py").read_text(encoding="utf-8")),
                  ("run_lab6.bat", (LAB / "run_lab6.bat").read_text(encoding="utf-8"))])

rep.save(OUT)
print("saved", OUT)

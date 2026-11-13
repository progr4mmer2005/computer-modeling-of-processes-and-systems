import contextlib
import io
import runpy

import numpy as np

from report_kit import ROOT, Report, console_png, num

LAB = ROOT / "lab8_python"
RES = LAB / "results"
OUT = ROOT / "Отчет_лабораторная_8_вариант_12.docx"

buffer = io.StringIO()
with contextlib.redirect_stdout(buffer):
    v = runpy.run_path(str(LAB / "lab8.py"), run_name="__main__")
lines = buffer.getvalue().splitlines()
params, peaks, variant = v["params"], v["peaks"], v["variant"]
terms, out_names = v["terms"], v["out_names"]


def p_text(p):
    return "(" + "; ".join(num(x, 4).rstrip("0").rstrip(",") if x else "0" for x in p) + ")"


rep = Report(8, "Система нечёткого вывода")

rep.h1("Цель работы")
rep.p("Изучить нечёткий логический вывод по алгоритму Мамдани, разработать систему нечёткого вывода и "
      "исследовать влияние функций принадлежности на результат.")

rep.h1("Задание")
rep.p(f"Вариант 12, тема 3, задание № {variant}:")
rep.numbered([
    "реализовать систему нечёткого вывода по примеру оценки состояния экосистемы и исследовать зависимость "
    "результата от параметров функций принадлежности;",
    "смоделировать систему управления холодопроизводительностью кондиционера MITSUBISHI HEAVY SRK25ZJP-S;",
    f"сделать выводы по функциям принадлежности выходной переменной варианта {variant}.",
])

rep.h1("Решение")
rep.h2("1. Алгоритм Мамдани")
rep.p("Для входов x_{1}, x_{2} и правила «ЕСЛИ x_{1} есть A_{k} И x_{2} есть B_{k} ТО y есть C_{k}»:")
rep.formula("α_{k} = min(μ_{Ak}(x_{1}), μ_{Bk}(x_{2})),")
rep.formula("μ′_{k}(y) = min(α_{k}, μ_{Ck}(y)),   μ(y) = max_{k} μ′_{k}(y),")
rep.formula("y* = Σ y·μ(y) / Σ μ(y),")
rep.p("где μ – степень принадлежности, α_{k} – степень истинности условия правила k, μ′_{k} – усечённый "
      "вывод правила, μ(y) – объединение выводов, y* – чёткий результат (центр тяжести). Сумма берётся "
      "по точкам сетки значений выхода.")

rep.h2("2. Состояние экосистемы")
rep.p("Входы: P-air (воздух), P-snow (снег), P-ground (почва), P-bio (биосфера), выход – состояние "
      "экосистемы. Диапазон всех переменных [0; 1], термы: «отличное», «хорошее», «напряжённое», "
      "«плохое». Типы функций принадлежности: zmf, pimf, pimf, smf:")
rep.formula("smf(x; a, b) = 2((x − a)/(b − a))^{2} при a ≤ x ≤ (a + b)/2;  1 − 2((x − b)/(b − a))^{2} "
            "при (a + b)/2 ≤ x ≤ b,")
rep.formula("zmf(x; a, b) = 1 − smf(x; a, b),   pimf(x; a, b, c, d) = min(smf(x; a, b), zmf(x; c, d)).")
rep.p("Исследованы три набора параметров (таблица 1). Набор «узкие» содержит параметры терма «хорошее» "
      "из примера (0,329; 0,392; 0,4034; 0,469).")
rep.table(["Набор", "отличное, zmf", "хорошее, pimf", "напряжённое, pimf", "плохое, smf"],
          [[k] + [p_text(x) for x in p] for k, p in params.items()],
          "Параметры функций принадлежности", widths=[2.0, 2.8, 4.2, 3.8, 2.6], size=8)
counts = [sum(1 for _, o in v["rules"] if o == k) for k in range(4)]
rep.p(f"База правил – все {len(v['rules'])} сочетаний термов четырёх входов. Термам присвоены номера 0–3; "
      "выходной терм равен среднему номеров входных термов, округлённому до целого. Пример: «ЕСЛИ P-air "
      "отличное И P-snow отличное И P-ground отличное И P-bio хорошее ТО состояние отличное» "
      f"(среднее 0,25 → 0). Правил по выходным термам: {', '.join(f'{t} {c}' for t, c in zip(terms, counts))}.")
mu = v["mu_air"]
rep.p(f"Расчёт для входа [0,5 0,5 0,5 0,5] (базовые параметры): μ_{{хорошее}}(0,5) = {num(mu[1], 3)}, "
      f"μ_{{напряжённое}}(0,5) = {num(mu[2], 3)}, остальные 0. Сила выходных термов «хорошее» и "
      f"«напряжённое» – {num(v['strength'][1], 3)}, результат y* = {num(v['value'], 3)}.")
rep.table(["Вход"] + list(params),
          [["[" + " ".join(num(a, 1) for a in x) + "]"] + [num(r, 3) for r in row]
           for x, row in v["eco_table"].items()],
          "Состояние экосистемы при разных параметрах функций принадлежности", widths=[4.5, 2.5, 2.5, 2.5])
rep.figure(RES / "02_eco_params.png", "Зависимость выхода от P-air (остальные входы 0,5)", 13.0)
rep.figure(RES / "03_eco_surface.png", "Поверхность вывода: P-air, P-snow (P-ground = P-bio = 0,5)", 12.0)

rep.h2("3. Кондиционер")
p = peaks[variant]
rep.p("Входы: разность температур Δt (0–35 К) и скорость изменения температуры V_{t} (0–0,35 К/мин), "
      "термы «малая», «средняя», «большая», треугольные функции с вершинами 0, 15, 30 К и 0; 0,15; 0,3 К/мин. "
      f"Выход: частота вращения компрессора f (0–150 Гц), пять термов варианта {variant} с вершинами "
      + ", ".join(f"{n} {num(x, 1).replace(',0', '')}" for n, x in zip(out_names, p)) +
      " Гц. Каждый треугольник начинается и заканчивается в вершинах соседних термов.")
rep.table(["№", "V_{t}", "Δt", "f"], [[k, a, b, c] for k, (a, b, c) in enumerate(v["ac_rules"], 1)],
          "Нечёткие правила", widths=[1.0, 3.0, 3.0, 3.5])
dt0, vt0, fired = v["dt0"], v["vt0"], v["fired0"]
mdt = {k: float(v["line"](dt0, *t)) for k, t in v["dt_terms"].items()}
mvt = {k: float(v["line"](vt0, *t)) for k, t in v["vt_terms"].items()}
rep.p(f"Пример: Δt = {num(dt0, 0)} К, V_{{t}} = {num(vt0, 2)} К/мин. Фаззификация:")
rep.formula(f"μ_{{средняя}}(20) = (30 − 20)/(30 − 15) = {num(mdt['средняя'], 3)},   "
            f"μ_{{большая}}(20) = (20 − 15)/(30 − 15) = {num(mdt['большая'], 3)};", numbered=False)
rep.formula(f"μ_{{малая}}(0,1) = (0,15 − 0,1)/0,15 = {num(mvt['малая'], 3)},   "
            f"μ_{{средняя}}(0,1) = 0,1/0,15 = {num(mvt['средняя'], 3)}.", numbered=False)
active = [(k, r, w) for k, (r, w) in enumerate(zip(v["ac_rules"], fired), 1) if w > 0]
rep.p("Сработавшие правила: " + "; ".join(
    f"{k}) min({num(mvt[a], 3)}; {num(mdt[b], 3)}) = {num(w, 3)} → f {c}" for k, (a, b, c), w in active) + ".")
agg, f = v["agg0"], v["f"]
rep.formula(f"f* = Σ f·μ(f) / Σ μ(f) = {num((f * agg).sum(), 1)} / {num(agg.sum(), 2)} = {num(v['f0'], 2)} Гц.",
            numbered=False)
rep.figure(RES / "05_ac_example.png", f"Объединение выводов правил и центр тяжести для Δt = 20 К, V_{{t}} = 0,1 К/мин",
           14.0)
rep.table(["V_{t} \\ Δt"] + [str(d) for d in v["dts"]],
          [[num(vt, 2)] + [num(x, 1) for x in row] for vt, row in zip(v["vts"], v["ac_table"])],
          "Частота вращения компрессора f, Гц", size=8)
rep.figure(RES / "06_ac_surface.png", "Поверхность вывода кондиционера", 12.0)

rep.h2("4. Сравнение вариантов функций принадлежности")
rows = []
for k, (vals, lo, hi) in v["variant_rows"].items():
    rows.append([f"{k}{' (наш)' if k == variant else ''}", ", ".join(num(x, 1).replace(",0", "") for x in peaks[k])]
                + [num(x, 1) for x in vals] + [f"{num(lo, 1)}–{num(hi, 1)}"])
rep.table(["Вариант", "Вершины, Гц"] + [f"Δt={d}, V_{{t}}={num(s, 2)}" for d, s in v["points"]] + ["Диапазон f"],
          rows, "Частота f, Гц, для четырёх вариантов выхода", widths=[1.8, 3.4, 2.0, 2.0, 2.0, 2.0, 2.4], size=8)
rep.figure(RES / "07_variants.png", "Частота при Δt = 20 К для четырёх вариантов", 12.0)

rep.h2("5. Вывод программы")
s2 = next(i for i, s in enumerate(lines) if s.startswith("2. КОНДИЦИОНЕР"))
shots = [(lines[:s2 - 1], "экосистема"), (lines[s2:], "кондиционер и сравнение вариантов")]
for k, (chunk, caption) in enumerate(shots, 1):
    png = RES / f"console_{k}.png"
    console_png(chunk, png, title="run_lab8.bat", font_size=26, light=True)
    rep.figure(png, "Вывод программы lab8.py: " + caption, 16.0)

et = v["eco_table"]
mid = et[(0.5, 0.5, 0.5, 0.5)]
lo_v, hi_v = v["variant_rows"][variant][1:]
rep.h1("Выводы")
rep.p(f"1. Реализована система нечёткого вывода Мамдани для экосистемы (4 входа, 256 правил). Результат зависит "
      f"от параметров функций принадлежности: для входа [0,5 0,5 0,5 0,5] он равен {num(mid[0], 3)}, "
      f"{num(mid[1], 3)} и {num(mid[2], 3)} для узких, базовых и широких термов. Узкие термы с промежутками "
      f"между ними делают выход почти постоянным; перекрывающиеся термы дают плавную зависимость.")
rep.p(f"2. Модель кондиционера даёт частоту компрессора от {num(lo_v, 1)} до {num(hi_v, 1)} Гц. Частота растёт "
      f"при увеличении разности температур и скорости её изменения; для Δt = 20 К, V_{{t}} = 0,1 К/мин "
      f"f = {num(v['f0'], 1)} Гц.")
rep.p(f"3. В варианте {variant} вершины выходных термов (1,5; 35; 68; 101; 134 Гц) расположены с шагом около 33 Гц и "
      "сдвинуты вправо относительно вариантов 1–3, поэтому при тех же входах частота выше, а диапазон выхода "
      "наибольший.")

rep.structural("Приложение А")
rep.p("Листинг программы", align="center", indent=False)
rep.code_columns([("lab8.py", (LAB / "lab8.py").read_text(encoding="utf-8")),
                  ("run_lab8.bat", (LAB / "run_lab8.bat").read_text(encoding="utf-8"))])

rep.save(OUT)
print("saved", OUT)

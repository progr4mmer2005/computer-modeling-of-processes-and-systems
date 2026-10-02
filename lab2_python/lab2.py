from math import sqrt
from pathlib import Path
import json

SNOW = 0.154
HAIR_LIMIT = 25.0
FREE = -99.808653
WATER_LINEAR = -72106.427618
SNOW_LINEAR = -261.604422
WATER_POWER = 6.497413e13
SNOW_POWER = 49416.888897
WATER_SQRT = 5434.970295
SNOW_SQRT = 106.466334
LEFT, RIGHT = 0.003, 0.005
EPS_VALUES = (1e-4, 1e-6, 1e-8, 1e-12)
REFERENCE_EPS = 1e-16
MAX_ITERATIONS = 200
GRAPH_STEP = 0.00005
CONSTANT_DIGITS = 9
RESULT_DIR = Path(__file__).resolve().parent / "results"


def constant_terms():
    return [FREE, SNOW_LINEAR * SNOW, SNOW_POWER * SNOW**5,
            SNOW_SQRT * sqrt(SNOW), -HAIR_LIMIT]


RAW_CONSTANT = sum(constant_terms())
CONSTANT = round(RAW_CONSTANT, CONSTANT_DIGITS)


def f(x):
    if x < 0:
        raise ValueError("Концентрация должна быть неотрицательной")
    return WATER_POWER * x**5 + WATER_LINEAR * x + WATER_SQRT * sqrt(x) + CONSTANT


def g(y):
    if y < 0:
        raise ValueError("При замене y = sqrt(x) требуется y >= 0")
    return WATER_POWER * y**10 + WATER_LINEAR * y**2 + WATER_SQRT * y + CONSTANT


def hair(x):
    return WATER_POWER * x**5 + WATER_LINEAR * x + WATER_SQRT * sqrt(x) + RAW_CONSTANT + HAIR_LIMIT


def bisect(function, left, right, eps):
    if not (left < right and eps > 0):
        raise ValueError("Нужны left < right и eps > 0")
    fl, fr = function(left), function(right)
    if fl == 0 or fr == 0:
        return (left if fl == 0 else right), 0
    if fl * fr > 0:
        raise ValueError("На границах интервала нет смены знака")
    for iteration in range(MAX_ITERATIONS + 1):
        middle = (left + right) / 2
        fm = function(middle)
        if (right - left) / 2 <= eps or fm == 0:
            return middle, iteration
        if middle == left or middle == right:
            raise ArithmeticError("Достигнут предел точности float")
        if fl * fm < 0:
            right = middle
        else:
            left, fl = middle, fm
    raise ArithmeticError("Превышено допустимое число итераций")


def ascii_table(headers, rows):
    rows = [[str(value) for value in row] for row in rows]
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    line = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    def row_text(row):
        return "| " + " | ".join(value.ljust(w) for value, w in zip(row, widths)) + " |"
    return "\n".join([line, row_text(headers), line, *map(row_text, rows), line])


def plots(root, transformed_root, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    xs = [i * GRAPH_STEP for i in range(round(RIGHT / GRAPH_STEP) + 1)]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, [f(x) for x in xs], label="f(x)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvspan(LEFT, RIGHT, color="orange", alpha=0.12, label="Интервал поиска")
    ax.scatter([root], [0], color="red", zorder=5, label=f"Корень {root:.12f}")
    ax.set(xlabel="Медь в питьевой воде x, мг/л", ylabel="f(x), мкг/г",
           title="Исходное уравнение после подстановки содержания меди в снеге")
    ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
    fig.savefig(RESULT_DIR / "01_function.png", dpi=200); plt.close(fig)
    ys = [sqrt(LEFT) + (sqrt(RIGHT) - sqrt(LEFT)) * i / 400 for i in range(401)]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ys, [g(y) for y in ys], label="g(y)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.scatter([transformed_root], [0], color="red", zorder=5,
               label=f"y = {transformed_root:.12f}")
    ax.set(xlabel="y = sqrt(x), sqrt(мг/л)", ylabel="g(y), мкг/г",
           title="Полином десятой степени после замены y = sqrt(x)")
    ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
    fig.savefig(RESULT_DIR / "02_polynomial.png", dpi=200); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.loglog([r["eps"] for r in results],
              [abs(r["root"] - root) for r in results], "o-", label="|xε − xопорн|")
    ax.loglog(EPS_VALUES, EPS_VALUES, "--", label="Граница ε")
    ax.invert_xaxis()
    ax.set(xlabel="Допустимая погрешность ε, мг/л", ylabel="Разность корней, мг/л",
           title="Уточнение результата методом половинного деления")
    ax.grid(which="both", alpha=0.3); ax.legend(); fig.tight_layout()
    fig.savefig(RESULT_DIR / "03_accuracy.png", dpi=200); plt.close(fig)


def main():
    RESULT_DIR.mkdir(exist_ok=True)
    root, reference_iterations = bisect(f, LEFT, RIGHT, REFERENCE_EPS)
    c, _ = bisect(g, sqrt(LEFT), sqrt(RIGHT), REFERENCE_EPS)
    d = c**2
    results = []
    for eps in EPS_VALUES:
        x, iterations = bisect(f, LEFT, RIGHT, eps)
        results.append(dict(eps=eps, root=x, iterations=iterations, residual=f(x)))
    rows = [[f'{r["eps"]:.0e}', f'{r["root"]:.15f}', r["iterations"],
             f'{r["residual"]:+.6e}', f'{abs(r["root"]-root):.3e}'] for r in results]
    table = ascii_table(["eps (mg/L)", "x (mg/L)", "steps", "f(x)", "|x-x_ref|"], rows)
    text = "\n".join([
        "Лабораторная работа 2. Вариант 13, задание 3. Медь.",
        f"Снег: {SNOW} мг/л; предел в волосах: {HAIR_LIMIT:g} мкг/г.",
        f"Постоянные слагаемые: {constant_terms()}",
        f"Постоянная до округления: {RAW_CONSTANT:.14f}",
        f"Постоянная для сравнения с Mathcad: {CONSTANT:.9f}",
        f"f({LEFT}) = {f(LEFT):.9f}; f({RIGHT}) = {f(RIGHT):.9f}",
        "Критерий остановки: (right-left)/2 <= eps.", table,
        f"Опорный корень: {root:.15f} мг/л ({reference_iterations} делений)",
        f"Невязка опорного корня: {f(root):+.6e}",
        f"В микрограммах на литр: {root*1000:.12f} мкг/л",
        f"Медь в волосах по полной исходной модели: {hair(root):.12f} мкг/г",
        f"y = {c:.15f}; x = y^2 = {d:.15f}; f(x) = {f(d):+.6e}",
        f"Разность решений через f и g: {abs(root-d):.6e} мг/л",
        f"f(0.003945904) = {f(0.003945904):+.6e}",
        f"f(0.003945904353) = {f(0.003945904353):+.6e}",
        "Невязка вычисляется до округления корня для вывода."])
    (RESULT_DIR / "results.txt").write_text(text + "\n", encoding="utf-8")
    payload = dict(constant=CONSTANT, raw_constant=RAW_CONSTANT, root=root,
                   residual=f(root), hair=hair(root), y=c, transformed_x=d,
                   transformed_residual=f(d), results=results, ascii_table=table)
    (RESULT_DIR / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(text)
    plots(root, c, results)
    print(f"\nТаблица, данные и три графика сохранены: {RESULT_DIR}")


if __name__ == "__main__":
    main()

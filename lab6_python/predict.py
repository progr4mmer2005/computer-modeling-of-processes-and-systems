from pathlib import Path

import numpy as np

path = Path(__file__).parent / "results" / "network.npz"
if not path.exists():
    print("Файл results/network.npz не найден. Сначала запустите run_lab6.bat, чтобы обучить и сохранить сеть.")
    input("Нажмите Enter для выхода...")
    raise SystemExit

net = np.load(path)
W1, b1, W2, b2 = net["W1"], net["b1"], net["W2"], net["b2"]
x_min, x_max, y_min, y_max = net["x_min"], net["x_max"], float(net["y_min"]), float(net["y_max"])

names = ["x1 - процент домохозяйств с дефицитом, %",
         "x2 - средний дефицит, руб.",
         "x3 - средний префицит, руб.",
         "x4 - мат. ожидание распределения, руб.",
         "x5 - дисперсия распределения, млн руб.²"]
example = [35.92, 7411.26, 15955.12, 15674.05, 53.93]


def predict(x):
    xn = (np.array(x) - x_min) / (x_max - x_min)
    h = 1 / (1 + np.exp(-(xn @ W1 + b1)))
    return float((h @ W2 + b2)[0]) * (y_max - y_min) + y_min


print("РАСЧЁТ СРЕДНЕГО ДОХОДА ДОМОХОЗЯЙСТВА СОХРАНЁННОЙ НЕЙРОСЕТЬЮ 5-8-1")
print("Сеть загружена из results/network.npz")
print("\nПример (средние значения по 43 районам):")
for name, value in zip(names, example):
    print(f"  {name:45} {value:12.2f}")
print(f"  Ответ сети: средний доход = {predict(example):.1f} руб.")

print("\nДиапазоны обучающих данных (лучше вводить значения внутри них):")
for name, low, high in zip(names, x_min, x_max):
    print(f"  {name:45} от {low:10.2f} до {high:10.2f}")

print("\nВведите 5 значений. Дробную часть можно писать через точку или запятую.")
print("Пустая строка или q - выход.")
while True:
    print()
    values = []
    for name, low, high in zip(names, x_min, x_max):
        while True:
            text = input(f"  {name}: ").strip().replace(",", ".")
            if text.lower() in ("", "q", "й"):
                raise SystemExit
            try:
                value = float(text)
                break
            except ValueError:
                print("    Нужно число, например 35.9")
        if not low <= value <= high:
            print(f"    Значение вне диапазона обучения ({low:.2f}...{high:.2f}), ответ может быть неточным")
        values.append(value)
    print(f"  Ответ сети: средний доход = {predict(values):.1f} руб.")

import math
from pathlib import Path

import numpy as np

path = Path(__file__).parent / "results" / "network.npz"

names = ["x1 - процент домохозяйств с дефицитом, %",
         "x2 - средний дефицит, руб.",
         "x3 - средний префицит, руб.",
         "x4 - мат. ожидание распределения, руб.",
         "x5 - дисперсия распределения, млн руб.²"]
limits = [(0, 100), (0, None), (0, None), (0, None), (0, None)]
example = [35.92, 7411.26, 15955.12, 15674.05, 53.93]
QUIT = ("q", "й", "выход", "exit")


def stop(message=""):
    if message:
        print(message)
    try:
        input("Нажмите Enter для выхода...")
    except (EOFError, KeyboardInterrupt):
        pass
    raise SystemExit


def ask(prompt):
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        stop("Ввод прерван.")


def read_value(name, allowed, trained):
    while True:
        text = ask(f"  {name}: ")
        if text.lower() in QUIT:
            stop("Работа завершена.")
        if not text:
            print("    Пусто. Введите число (или q для выхода).")
            continue
        text = text.replace(" ", "").replace(" ", "").replace(",", ".")
        try:
            value = float(text)
        except ValueError:
            print(f"    «{text}» – не число. Пример правильного ввода: 35.9 или 35,9")
            continue
        if math.isnan(value) or math.isinf(value):
            print("    Нужно обычное конечное число.")
            continue
        low, high = allowed
        if low is not None and value < low:
            print(f"    Значение не может быть меньше {low}.")
            continue
        if high is not None and value > high:
            print(f"    Значение не может быть больше {high}.")
            continue
        if not trained[0] <= value <= trained[1]:
            print(f"    Вне диапазона обучающих данных ({trained[0]:.2f} ... {trained[1]:.2f}), "
                  f"ответ сети может быть неточным.")
            answer = ask("    Оставить это значение? (д - да, любая другая клавиша - ввести заново): ").lower()
            if answer not in ("д", "да", "y", "yes", "l"):
                continue
        return value


if not path.exists():
    stop("Файл results/network.npz не найден. Сначала запустите run_lab6.bat, чтобы обучить и сохранить сеть.")
try:
    net = np.load(path)
    W1, b1, W2, b2 = net["W1"], net["b1"], net["W2"], net["b2"]
    x_min, x_max = net["x_min"], net["x_max"]
    y_min, y_max = float(net["y_min"]), float(net["y_max"])
    if W1.shape != (5, W2.shape[0]) or x_min.shape != (5,):
        raise ValueError("неверные размеры массивов")
except Exception as error:
    stop(f"Не удалось прочитать файл сети ({error}). Запустите run_lab6.bat заново.")


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

print("\nВведите 5 значений по одному. Дробную часть можно писать через точку или запятую.")
print("Для выхода в любой момент введите q.")
count = 0
while True:
    count += 1
    print(f"\nРасчёт {count}")
    values = [read_value(name, allowed, (low, high))
              for name, allowed, low, high in zip(names, limits, x_min, x_max)]
    print(f"  Ответ сети: средний доход = {predict(values):.1f} руб.")
    if ask("\nЕщё один расчёт? (Enter - да, q - выход): ").lower() in QUIT:
        stop("Работа завершена.")

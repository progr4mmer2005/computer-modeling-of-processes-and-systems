"""Simple queue simulation for Lab 5, Topic 3, Variant 12."""
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Time is measured in minutes.
WORKDAY = 8 * 60
REPETITIONS = 10_000
SEED = random.SystemRandom().randrange(1, 2**32)
CASHIERS = 2
ATM_SERVERS = 1
CASHIER_QUEUE_LIMIT = 20
ATM_QUEUE_LIMIT = 15

# Each tuple is (minimum, maximum, most likely value).
ARRIVAL_TIME = (1.0, 3.0, 2.0)
CASHIER_TIME = (5.0, 9.0, 7.0)
ATM_TIME = (0.8, 1.3, 1.0)

OUTPUT = Path(__file__).parent / "results" / "results.txt"
OUTPUT.parent.mkdir(exist_ok=True)


def random_time(generator, limits):
    return generator.triangular(limits[0], limits[1], limits[2])


def make_station(server_count, queue_limit, service_limits, keep_trace=False):
    return {
        "free_at": [0.0] * server_count,
        "customer_arrival": [None] * server_count,
        "queue": [],
        "queue_limit": queue_limit,
        "service_limits": service_limits,
        "last_time": 0.0,
        "queue_area": 0.0,
        "busy_time": 0.0,
        "max_queue": 0,
        "arrived": 0,
        "refused": 0,
        "waits": [],
        "system_times": [],
        "keep_trace": keep_trace,
        "queue_trace": [(0.0, 0)] if keep_trace else None,
    }


def start_service(station, server, arrival, start, generator):
    service = random_time(generator, station["service_limits"])
    finish = start + service
    station["free_at"][server] = finish
    station["customer_arrival"][server] = arrival
    station["waits"].append(start - arrival)
    station["busy_time"] += max(0.0, min(finish, WORKDAY) - min(start, WORKDAY))


def update_station(station, until, generator):
    """Finish work up to the next arrival and start customers already waiting."""
    while True:
        server = None
        for i, arrival in enumerate(station["customer_arrival"]):
            if arrival is not None and station["free_at"][i] <= until:
                if server is None or station["free_at"][i] < station["free_at"][server]:
                    server = i
        if server is None:
            break

        finish = station["free_at"][server]
        area_end = min(finish, WORKDAY)
        area_start = min(station["last_time"], WORKDAY)
        if area_end > area_start:
            station["queue_area"] += len(station["queue"]) * (area_end - area_start)
        station["last_time"] = finish

        arrival = station["customer_arrival"][server]
        station["system_times"].append(finish - arrival)
        station["customer_arrival"][server] = None
        if station["queue"]:
            next_arrival = station["queue"].pop(0)
            start_service(station, server, next_arrival, finish, generator)
        if station["keep_trace"]:
            station["queue_trace"].append((finish, len(station["queue"])))

    if until != float("inf"):
        area_end = min(until, WORKDAY)
        area_start = min(station["last_time"], WORKDAY)
        if area_end > area_start:
            station["queue_area"] += len(station["queue"]) * (area_end - area_start)
        station["last_time"] = until


def send_customer(station, now, generator):
    station["arrived"] += 1
    free_server = next((i for i, arrival in enumerate(station["customer_arrival"])
                        if arrival is None), None)
    if free_server is not None:
        start_service(station, free_server, now, now, generator)
    elif len(station["queue"]) < station["queue_limit"]:
        station["queue"].append(now)
        station["max_queue"] = max(station["max_queue"], len(station["queue"]))
    else:
        station["refused"] += 1
    if station["keep_trace"]:
        station["queue_trace"].append((now, len(station["queue"])))


def simulate(seed, keep_trace=False):
    generator = random.Random(seed)
    cashier = make_station(CASHIERS, CASHIER_QUEUE_LIMIT, CASHIER_TIME, keep_trace)
    atm = make_station(ATM_SERVERS, ATM_QUEUE_LIMIT, ATM_TIME, keep_trace)
    stations = {"cashier": cashier, "atm": atm}

    arrival = random_time(generator, ARRIVAL_TIME)
    while arrival < WORKDAY:
        # Variant 12 routes customers to cashiers and ATM in a 1:2 ratio.
        channel = "cashier" if generator.random() < 1 / 3 else "atm"
        station = stations[channel]
        update_station(station, arrival, generator)
        send_customer(station, arrival, generator)
        arrival += random_time(generator, ARRIVAL_TIME)

    # Let the customers who arrived before closing finish their service.
    for station in stations.values():
        update_station(station, float("inf"), generator)

    result = {}
    for name, station in stations.items():
        served = len(station["system_times"])
        result[name] = {
            "arrived": station["arrived"],
            "served": served,
            "refused": station["refused"],
            "wait": sum(station["waits"]) / served if served else 0.0,
            "system": sum(station["system_times"]) / served if served else 0.0,
            "queue": station["queue_area"] / WORKDAY,
            "max_queue": station["max_queue"],
            "utilization": station["busy_time"] / (len(station["free_at"]) * WORKDAY),
        }
    if keep_trace:
        result["trace"] = {name: station["queue_trace"] for name, station in stations.items()}
    return result


def save_graphs(averages, queued_runs, trace):
    """Save simple plots based on the simulation results."""
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 180, "font.size": 10,
                         "axes.grid": True, "grid.alpha": 0.3})

    fig, ax = plt.subplots(figsize=(8, 4.2))
    for name, label in (("cashier", "Очередь к кассирам"), ("atm", "Очередь к банкомату")):
        points = trace[name]
        ax.step([p[0] for p in points], [p[1] for p in points], where="post", label=label)
    ax.set(xlim=(0, WORKDAY), xlabel="Модельное время, мин", ylabel="Человек в очереди",
           title="Длина очереди в одном типичном прогоне")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT.parent / "01_queues_over_time.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
    labels = ["Кассиры", "Банкомат"]
    axes[0].bar(labels, [averages[n]["wait"] for n in ("cashier", "atm")], color=["#4472C4", "#ED7D31"])
    axes[0].set(ylabel="Минуты", title="Среднее ожидание")
    axes[1].bar(labels, [averages[n]["queue"] for n in ("cashier", "atm")], color=["#4472C4", "#ED7D31"])
    axes[1].set(ylabel="Человек", title="Средняя длина очереди")
    fig.tight_layout()
    fig.savefig(OUTPUT.parent / "02_wait_and_queue.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    utilization = [100 * averages[n]["utilization"] for n in ("cashier", "atm")]
    ax.bar(labels, utilization, color=["#4472C4", "#ED7D31"])
    ax.set(ylabel="Загрузка, %", title="Средняя загрузка приборов", ylim=(0, 100))
    for i, value in enumerate(utilization):
        ax.text(i, value + 1, f"{value:.1f}%", ha="center")
    fig.tight_layout()
    fig.savefig(OUTPUT.parent / "03_utilization.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    probability = [100 * queued_runs[n] / REPETITIONS for n in ("cashier", "atm")]
    ax.bar(labels, probability, color=["#4472C4", "#ED7D31"])
    ax.set(ylabel="Прогонов с очередью, %", title="Как часто возникала очередь", ylim=(0, 110))
    for i, value in enumerate(probability):
        ax.text(i, value + 1, f"{value:.1f}%", ha="center")
    fig.tight_layout()
    fig.savefig(OUTPUT.parent / "04_queue_frequency.png")
    plt.close(fig)


def main():
    totals = {name: {key: 0.0 for key in
                     ("arrived", "served", "refused", "wait", "system", "queue", "max_queue", "utilization")}
              for name in ("cashier", "atm")}

    queued_runs = {name: 0 for name in totals}
    example_trace = None
    for repetition in range(REPETITIONS):
        result = simulate(SEED + repetition, keep_trace=(repetition == 0))
        if repetition == 0:
            example_trace = result["trace"]
        for name in totals:
            for key in totals[name]:
                totals[name][key] += result[name][key]
            if result[name]["max_queue"] > 0:
                queued_runs[name] += 1

    averages = {name: {key: value / REPETITIONS for key, value in totals[name].items()}
                for name in totals}
    save_graphs(averages, queued_runs, example_trace)
    output = [
        "ЛР5. Модель банковского отделения, вариант 12",
        f"Запусков: {REPETITIONS}; каждый запуск моделирует один рабочий день ({WORKDAY} минут).",
        f"Случайный начальный seed этого расчёта: {SEED}.",
        "Кассиры: 2; банкомат: 1; клиенты направляются к ним в соотношении 1:2.",
        "Приход: triangular(1, 2, 3); кассир: triangular(5, 7, 9); банкомат: triangular(0.8, 1.0, 1.3).",
        "Средние результаты за один день по всем запускам:",
        "Канал     Пришло  Обслужено  Отказано  Ожидание, мин  Очередь  Макс. очередь  Загрузка",
    ]
    for name, label in (("cashier", "Кассиры"), ("atm", "Банкомат")):
        values = averages[name]
        output.append(
            f"{label:<9} {values['arrived']:>7.1f} {values['served']:>10.1f} {values['refused']:>9.3f}"
            f" {values['wait']:>14.2f} {values['queue']:>8.2f} {values['max_queue']:>14.2f}"
            f" {100 * values['utilization']:>7.1f}%"
        )
    output.append(f"Всего обслужено в среднем: {averages['cashier']['served'] + averages['atm']['served']:.1f} клиентов за день.")
    output.append(f"Очередь возникала в прогонах: кассиры {100 * queued_runs['cashier'] / REPETITIONS:.1f}%, "
                  f"банкомат {100 * queued_runs['atm'] / REPETITIONS:.1f}%.")
    output.append("Параметры распределений выбраны для задания; при их изменении изменятся и результаты.")
    output.append("Графики PNG сохранены в папке results: 01_queues_over_time.png, 02_wait_and_queue.png, 03_utilization.png, 04_queue_frequency.png.")

    text = "\n".join(output) + "\n"
    print(text, end="")
    OUTPUT.write_text(text, encoding="utf-8")
    print(f"Текстовый итог сохранен в: {OUTPUT}")


if __name__ == "__main__":
    main()

"""Build the concise report for Lab 5, Topic 3, Variant 12."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from report_kit import Report
from docx.shared import Pt

LAB = ROOT / "lab5_python"
RESULT = LAB / "results" / "results.txt"
OUT = ROOT / "Отчет_лабораторная_5_вариант_12_с_графиками.docx"

rep = Report(5, "Моделирование банковского отделения")
rep.h1("Цель работы")
rep.p("Смоделировать работу банковского отделения с очередями, кассирами и банкоматом.")

rep.h1("Задание и параметры")
rep.p("Вариант 12: клиенты приходят в банк, примерно треть направляется к кассирам, остальные — к банкомату. В банке работают два кассира и один банкомат. Если все приборы выбранного канала заняты, клиент ждёт; если очередь заполнена, получает отказ.")
rep.table(["Что задаём", "Значение"], [
    ["Приход клиентов", "Треугольный интервал 1–3 мин, наиболее вероятно 2 мин"],
    ["Направление клиентов", "Кассир : банкомат = 1 : 2"],
    ["Обслуживание у кассира", "Треугольное время 5–9 мин, наиболее вероятно 7 мин"],
    ["Обслуживание у банкомата", "Треугольное время 0,8–1,3 мин, наиболее вероятно 1 мин"],
    ["Вместимость очередей", "20 мест к кассирам, 15 к банкомату"],
], "Параметры модели", widths=[4.3, 12.2], size=9)
rep.p("Численные параметры распределений методичка разрешает выбрать самостоятельно. Они заданы явно, чтобы расчёт можно было повторить.")

rep.h1("Как работает программа")
rep.p("Программа автоматически моделирует 10 000 отдельных восьмичасовых дней. Она переходит от одного прихода или окончания обслуживания к следующему, ведёт очереди и записывает показатели каждого дня. В конце выводит средние результаты по всем прогонам.")
rep.p("Ниже приведён фактический текстовый вывод программы.")
for line in RESULT.read_text(encoding="utf-8").splitlines():
    par = rep.doc.add_paragraph()
    par.paragraph_format.first_line_indent = 0
    par.paragraph_format.space_after = 0
    par.paragraph_format.keep_together = True
    run = par.add_run(line)
    run.font.name = "Consolas"
    run.font.size = Pt(8)

rep.h1("Графики и диаграммы")
rep.p("На первом графике показано изменение очередей в одном типичном прогоне. Остальные диаграммы показывают средние показатели или долю прогонов, где возникала очередь, по всем 10 000 запускам.")
rep.figure(LAB / "results" / "01_queues_over_time.png", "Длина очереди к кассирам и банкомату в одном прогоне", 15.5)
rep.figure(LAB / "results" / "02_wait_and_queue.png", "Среднее время ожидания и средняя длина очереди по 10 000 прогонам", 15.5)
rep.figure(LAB / "results" / "03_utilization.png", "Средняя загрузка кассиров и банкомата", 15.5)
rep.figure(LAB / "results" / "04_queue_frequency.png", "Доля прогонов, в которых хотя бы раз возникала очередь", 15.5)

rep.h1("Вывод")
rep.p("В среднем за день обслуживается около 240 клиентов, отказы практически отсутствуют. Среднее ожидание небольшое; очередь чаще возникает к кассирам, поскольку обслуживание у них дольше. Результаты усреднены по 10 000 прогонов.")
rep.p("Полный листинг программы приведён в приложении А.")

rep.structural("Приложение А")
rep.p("Листинг программы lab5.py", align="center", indent=False)
for code_line in (LAB / "lab5.py").read_text(encoding="utf-8").splitlines():
    par = rep.doc.add_paragraph()
    par.paragraph_format.first_line_indent = 0
    par.paragraph_format.space_before = 0
    par.paragraph_format.space_after = 0
    par.paragraph_format.line_spacing = Pt(8)
    par.paragraph_format.keep_together = True
    run = par.add_run(code_line or " ")
    run.font.name = "Consolas"
    run.font.size = Pt(6.5)

rep.save(OUT)
print("saved", OUT)

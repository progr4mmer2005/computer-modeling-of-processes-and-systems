# tools – скрипты для отчетов и проверки

| Файл | Что делает |
|------|------------|
| `report_kit.py` | Общий модуль сборки отчетов: шаблон диплома из `references/` (основной шрифт 10 pt), титульный лист из `references/титульный лист.docx`, заголовки, таблицы без заливки, рисунки и формулы с нумерацией, листинги кода в 3 колонки (Consolas 5 pt), картинки вывода консоли |
| `build_lab3_report.py` | Отчет по лабе 3 из `lab3_python/results/results.json` и рисунков |
| `build_lab4_report.py` | Отчет по лабе 4 из `lab4_python/results/results.json`, рисунков и картинок Excel |
| `docx_to_pdf.ps1` | Конвертация docx в PDF через Microsoft Word |
| `preview_docx.py` | Превью отчета: PDF и PNG всех страниц, листы-сводки по 3 страницы (папка `preview/`) |
| `pdf_to_images.py` | Рендер страниц PDF (методичек) в PNG, при `--text` – еще и текст |
| `check_html_links.py` | Проверка ссылок-якорей в HTML-шпаргалках из `защита/` |
| `fix_guide_css.py` | Исправление стилей блоков в HTML-шпаргалках (подпись блока на отдельной строке, рамка) |

## Порядок пересборки

```powershell
cd lab3_python; python -X utf8 lab3.py; cd ..
cd lab4_python; python -X utf8 lab4.py; cd ..
python -X utf8 tools/build_lab3_report.py
python -X utf8 tools/build_lab4_report.py
python tools/preview_docx.py Отчет_лабораторная_3_вариант_12.docx
python tools/check_html_links.py
```

Папка `references/` (методички, шаблон диплома, титульный лист) в репозиторий не входит
и должна лежать локально – из нее `report_kit.py` берет шаблоны.

Отчет нельзя пересобрать, пока он открыт в Word.
Для `preview_docx.py` и `docx_to_pdf.ps1` нужен установленный Microsoft Word.

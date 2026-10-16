"""Чинит стили блоков в HTML-шпаргалках: рамка и отдельная строка для подписи
(«Говори так», «А почему?», «Пример», «Ловушка») применяются к самим блокам.

Запуск: python tools/fix_guide_css.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = (".box { padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 16px 0; border-left: 4px solid; }\n"
       ".box > b.lbl { display: block; font-size: 12.5px; text-transform: uppercase; "
       "letter-spacing: .07em; margin-bottom: 4px; }")
NEW = (".say, .warn, .why, .ex { padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 16px 0;\n"
       "  border-left: 4px solid; }\n"
       "b.lbl { display: block; font-size: 12.5px; text-transform: uppercase; letter-spacing: .07em;\n"
       "  margin-bottom: 6px; }")

for path in sorted((ROOT / "защита").glob("*.html")):
    text = path.read_text(encoding="utf-8")
    if OLD in text:
        path.write_text(text.replace(OLD, NEW), encoding="utf-8")
        print("исправлен:", path.name)
    else:
        print("без изменений:", path.name)

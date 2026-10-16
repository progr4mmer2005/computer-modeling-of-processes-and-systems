"""Встраивает картинки в HTML-шпаргалки из защита/ (data URI), чтобы файл открывался
отдельно от проекта. Исходный путь хранится в data-src, поэтому после пересчета
графиков достаточно запустить скрипт еще раз.

Запуск: python tools/embed_guide_images.py
"""
import base64
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GUIDES = ROOT / "защита"
IMG = re.compile(r'<img ([^>]*?)(?<![\w-])src="([^"]+)"([^>]*)>')


def embed(match, base):
    before, src, after = match.groups()
    attrs = before + after
    original = re.search(r'data-src="([^"]+)"', attrs)
    path = original.group(1) if original else src
    if path.startswith("data:"):
        return match.group(0)
    data = base64.b64encode((base / path).resolve().read_bytes()).decode("ascii")
    attrs = re.sub(r'\s*data-src="[^"]+"', "", attrs).strip()
    return f'<img {attrs} data-src="{path}" src="data:image/png;base64,{data}">'


for page in sorted(GUIDES.glob("*.html")):
    text = page.read_text(encoding="utf-8")
    new, count = IMG.subn(lambda m: embed(m, page.parent), text)
    if new != text:
        page.write_text(new, encoding="utf-8")
    print(f"{page.name}: картинок {count}, размер {len(new.encode('utf-8')) // 1024} КБ")

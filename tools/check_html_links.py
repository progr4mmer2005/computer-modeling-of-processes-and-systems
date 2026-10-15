"""Проверяет, что все внутренние ссылки href="#id" в HTML-файлах ведут на существующие id.

Запуск: python tools/check_html_links.py [файлы.html ...]
Без аргументов проверяет все файлы в папке защита/.
"""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]


def check(path):
    text = path.read_text(encoding="utf-8")
    ids = set(re.findall(r'\bid="([^"]+)"', text))
    links = re.findall(r'href="#([^"]+)"', text)
    missing = sorted({link for link in links if link not in ids})
    unused = sorted(i for i in ids if i.startswith("g-") and i not in links)
    print(f"{path.name}: ссылок {len(links)}, якорей {len(ids)}, битых {len(missing)}")
    for link in missing:
        print("  нет якоря:", link)
    for anchor in unused:
        print("  термин без ссылок из текста:", anchor)
    return not missing


def main():
    files = [Path(a) for a in sys.argv[1:]] or sorted((ROOT / "защита").glob("*.html"))
    ok = all([check(f) for f in files])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

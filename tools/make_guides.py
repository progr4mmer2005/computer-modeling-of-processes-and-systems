import subprocess
import sys
from pathlib import Path

TOOLS = Path(__file__).parent
ROOT = TOOLS.parent
style = (TOOLS / "guide_style.css.html").read_text(encoding="utf-8")

for lab in (4, 6, 7, 8):
    body = (TOOLS / "guides" / f"lab{lab}_body.html").read_text(encoding="utf-8")
    page = ('<!doctype html>\n<html lang="ru">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f"<title>Защита лабы {lab}</title>\n{style}\n</head>\n<body>\n<main>\n{body}</main>\n</body>\n</html>\n")
    target = ROOT / "защита" / f"Защита_лаба_{lab}.html"
    target.write_text(page, encoding="utf-8")
    print("создан", target.name)

subprocess.run([sys.executable, str(TOOLS / "embed_guide_images.py")], check=True)

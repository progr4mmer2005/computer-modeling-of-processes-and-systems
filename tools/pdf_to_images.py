"""Рендерит страницы PDF в PNG (например, методички из references/, у которых не извлекается текст).

Запуск:
  python tools/pdf_to_images.py файл.pdf папка_вывода [--scale 1.4] [--text]
--text дополнительно сохраняет извлеченный текст страниц в файл <имя>.txt.
"""
from pathlib import Path
import argparse

import pypdfium2 as pdfium


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--scale", type=float, default=1.4)
    parser.add_argument("--text", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pdf = pdfium.PdfDocument(str(args.pdf))
    texts = []
    for i, page in enumerate(pdf, 1):
        page.render(scale=args.scale).to_pil().save(args.out / f"{args.pdf.stem}_p{i:02d}.png")
        if args.text:
            texts.append(f"===== PAGE {i}\n" + page.get_textpage().get_text_range())
    if args.text:
        (args.out / f"{args.pdf.stem}.txt").write_text("\n".join(texts), encoding="utf-8")
    print(f"{args.pdf.name}: {len(pdf)} страниц -> {args.out}")


if __name__ == "__main__":
    main()

"""Превью отчета: docx -> PDF через Microsoft Word (docx_to_pdf.ps1) -> PNG страниц и листы-сводки.

Запуск: python tools/preview_docx.py Отчет.docx [папка_вывода] [--per-sheet 3]
По умолчанию результат кладется в preview/<имя отчета>/ в корне проекта.
"""
from pathlib import Path
import argparse
import subprocess

import pypdfium2 as pdfium
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CONVERTER = Path(__file__).with_name("docx_to_pdf.ps1")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("docx", type=Path)
    parser.add_argument("out", type=Path, nargs="?")
    parser.add_argument("--per-sheet", type=int, default=3)
    parser.add_argument("--scale", type=float, default=0.9)
    args = parser.parse_args()
    docx_path = args.docx.resolve()
    out = (args.out or ROOT / "preview" / docx_path.stem).resolve()
    out.mkdir(parents=True, exist_ok=True)
    pdf_path = out / f"{docx_path.stem}.pdf"
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(CONVERTER),
                    "-Docx", str(docx_path), "-Pdf", str(pdf_path)], check=True)
    pdf = pdfium.PdfDocument(str(pdf_path))
    pages = []
    for i, page in enumerate(pdf, 1):
        file = out / f"p{i:02d}.png"
        page.render(scale=args.scale).to_pil().save(file)
        pages.append(file)
    pdf.close()
    per = args.per_sheet
    for k in range(0, len(pages), per):
        images = [Image.open(p) for p in pages[k:k + per]]
        w, h = images[0].size
        sheet = Image.new("RGB", (w * len(images) + 10 * (len(images) - 1), h), "gray")
        for j, im in enumerate(images):
            sheet.paste(im, (j * (w + 10), 0))
        sheet.save(out / f"sheet_{k // per + 1:02d}.png")
    print(f"{docx_path.name}: {len(pages)} страниц -> {out}")


if __name__ == "__main__":
    main()

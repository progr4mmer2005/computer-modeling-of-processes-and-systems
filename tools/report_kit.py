"""Общие функции сборки отчетов.

Основа - шаблон диплома из references (стили, поля, нумерация страниц);
базовый размер шрифта уменьшен до 10 pt. Первая страница - титульный лист
из references/титульный лист.docx с заполненными полями.
"""
from copy import deepcopy
from pathlib import Path
import re

import docx
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
REFERENCES = ROOT / "references"
TEMPLATE = next(REFERENCES.glob("Diplom_shablon*.docx"))
TITLE_PAGE = REFERENCES / "титульный лист.docx"

STUDENT = "Наумов Д.О."
GROUP = "4411"
DISCIPLINE = "Компьютерное моделирование процессов и систем"
BODY_PT = 10
TEXT_WIDTH_CM = 16.5

H1, H2, H3 = "Заголовок 1 (раздел)", "Заголовок 2 (подраздел)", "Заголовок 3 (пункт)"
FIGURE_CAPTION, TABLE_CAPTION = "Подпись рисунка", "Название таблицы"


def num(value, digits=4):
    """Число с десятичной запятой."""
    return f"{value:.{digits}f}".replace(".", ",").replace("-", "−")


def sci(value, digits=2):
    """Экспоненциальная запись вида 1,23·10⁻⁴."""
    if value == 0:
        return "0"
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return f"{mantissa.replace('.', ',').replace('-', '−')}·10{str(int(exponent)).translate(sup)}"


def smart(value, digits=4):
    """Обычная запись для умеренных чисел, экспоненциальная для очень больших/малых."""
    if value != 0 and (abs(value) >= 1e4 or abs(value) < 1e-3):
        return sci(value, 2)
    return num(value, digits)


def _set(parent, tag, **attrs):
    el = parent.find(qn(tag))
    if el is None:
        el = OxmlElement(tag)
        parent.append(el)
    for k, v in attrs.items():
        el.set(qn(k), str(v))
    return el


class Report:
    def __init__(self, lab_number, topic):
        self.doc = docx.Document(str(TEMPLATE))
        self.figures = 0
        self.tables = 0
        self.formulas = 0
        self._tune_styles()
        body = self.doc.element.body
        final_sect = body.find(qn("w:sectPr"))
        title_sect = deepcopy(next(s for s in body.iter(qn("w:sectPr")) if s is not final_sect))
        for child in list(body):
            if child is not final_sect:
                body.remove(child)
        self._title_page(lab_number, topic, final_sect)
        brk = OxmlElement("w:p")
        ppr = OxmlElement("w:pPr")
        ppr.append(title_sect)
        brk.append(ppr)
        final_sect.addprevious(brk)

    # ---------- оформление ----------

    def _tune_styles(self):
        styles = self.doc.styles.element
        rpr = styles.find(qn("w:docDefaults")).find(qn("w:rPrDefault")).find(qn("w:rPr"))
        _set(rpr, "w:sz", **{"w:val": BODY_PT * 2})
        _set(rpr, "w:szCs", **{"w:val": BODY_PT * 2})
        for name, before, after in ((H1, 14, 6), (H2, 10, 4), (H3, 8, 4)):
            pf = self.doc.styles[name].paragraph_format
            pf.page_break_before = False
            pf.space_before, pf.space_after = Pt(before), Pt(after)
            pf.keep_with_next = True
        self.doc.styles[H1].font.size = Pt(12)
        self.doc.styles[H2].font.size = Pt(11)

    def _title_page(self, lab_number, topic, anchor):
        source = docx.Document(str(TITLE_PAGE))
        for par in source.paragraphs:
            text = par.text
            el = deepcopy(par._p)
            ppr = el.find(qn("w:pPr"))
            if ppr is None:
                ppr = OxmlElement("w:pPr")
                el.insert(0, ppr)
            if ppr.find(qn("w:ind")) is None:
                _set(ppr, "w:ind", **{"w:firstLine": 0})
            else:
                ppr.find(qn("w:ind")).set(qn("w:firstLine"), "0")
            _set(ppr, "w:spacing", **{"w:before": 0, "w:after": 0, "w:line": 240,
                                      "w:lineRule": "auto"})
            if ppr.find(qn("w:jc")) is None:
                _set(ppr, "w:jc", **{"w:val": "center"})
            for r in el.iter(qn("w:r")):
                rpr = r.find(qn("w:rPr"))
                if rpr is None:
                    rpr = OxmlElement("w:rPr")
                    r.insert(0, rpr)
                if rpr.find(qn("w:sz")) is None:
                    _set(rpr, "w:sz", **{"w:val": 24})
            replacement = None
            if text.startswith("Лабораторная работа"):
                replacement = f"Лабораторная работа № {lab_number}"
            elif text.startswith("по дисциплине"):
                replacement = f"по дисциплине «{DISCIPLINE}»"
            elif text.strip().startswith("____") and len(text.strip("_ ")) == 0 and "Выполнил" not in text:
                replacement = topic if not getattr(self, "_topic_done", False) else None
                self._topic_done = True
            elif text.startswith("название темы"):
                replacement = ""
            elif text.startswith("студент группы"):
                replacement = f"студент группы {GROUP}"
            elif "(фио)" in text:
                replacement = STUDENT
            if replacement is not None:
                runs = list(el.iter(qn("w:r")))
                for r in runs[1:]:
                    r.getparent().remove(r)
                if runs:
                    for t in runs[0].findall(qn("w:t")):
                        runs[0].remove(t)
                    t = OxmlElement("w:t")
                    t.text = replacement
                    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
                    runs[0].append(t)
                    if replacement == topic:
                        _set(runs[0].find(qn("w:rPr")), "w:b")
            anchor.addprevious(el)

    # ---------- элементы текста ----------

    def _runs(self, par, text, size=None, bold=None):
        pattern = r"(\*\*.+?\*\*|_\{.+?\}|\^\{.+?\}|//.+?//)"
        for part in re.split(pattern, text):
            if not part:
                continue
            if part.startswith("**"):
                run = par.add_run(part[2:-2])
                run.bold = True
            elif part.startswith("_{"):
                run = par.add_run(part[2:-1])
                run.font.subscript = True
            elif part.startswith("^{"):
                run = par.add_run(part[2:-1])
                run.font.superscript = True
            elif part.startswith("//"):
                run = par.add_run(part[2:-2])
                run.italic = True
            else:
                run = par.add_run(part)
            if size:
                run.font.size = Pt(size)
            if bold:
                run.bold = True
        return par

    def h1(self, text):
        return self.doc.add_paragraph(text, style=H1)

    def h2(self, text):
        return self.doc.add_paragraph(text, style=H2)

    def h3(self, text):
        return self.doc.add_paragraph(text, style=H3)

    def p(self, text, align=None, indent=True, keep=False):
        par = self._runs(self.doc.add_paragraph(), text)
        if align == "center":
            par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if not indent:
            par.paragraph_format.first_line_indent = Cm(0)
        if keep:
            par.paragraph_format.keep_with_next = True
        return par

    def bullets(self, items):
        for item in items:
            par = self._runs(self.doc.add_paragraph(), "– " + item)
            par.paragraph_format.keep_together = True

    def numbered(self, items):
        for k, item in enumerate(items, 1):
            self._runs(self.doc.add_paragraph(), f"{k}) {item}")

    def formula(self, text, numbered=True):
        """Формула по центру, номер в круглых скобках у правого поля."""
        par = self.doc.add_paragraph()
        pf = par.paragraph_format
        pf.first_line_indent = Cm(0)
        pf.space_before = pf.space_after = Pt(2)
        pf.tab_stops.add_tab_stop(Cm(TEXT_WIDTH_CM / 2), WD_TAB_ALIGNMENT.CENTER)
        pf.tab_stops.add_tab_stop(Cm(TEXT_WIDTH_CM), WD_TAB_ALIGNMENT.RIGHT)
        par.add_run("\t")
        self._runs(par, text)
        if numbered:
            self.formulas += 1
            par.add_run(f"\t({self.formulas})")
        return self.formulas

    def figure(self, path, caption, width_cm=15.0):
        self.figures += 1
        par = self.doc.add_paragraph()
        par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        par.paragraph_format.first_line_indent = Cm(0)
        par.paragraph_format.keep_with_next = True
        par.paragraph_format.space_before = Pt(6)
        par.paragraph_format.line_spacing = 1.0
        par.add_run().add_picture(str(path), width=Cm(width_cm))
        self._runs(self.doc.add_paragraph(style=FIGURE_CAPTION),
                   f"Рисунок {self.figures} – {caption}")
        return self.figures

    def next_figure(self):
        return self.figures + 1

    def next_table(self):
        return self.tables + 1

    def table(self, headers, rows, caption, widths=None, size=9, align=None, bold_last=False):
        self.tables += 1
        cap = self._runs(self.doc.add_paragraph(style=TABLE_CAPTION),
                         f"Таблица {self.tables} – {caption}")
        cap.paragraph_format.keep_with_next = True
        table = self.doc.add_table(rows=1 + len(rows), cols=len(headers))
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        tbl_pr = table._tbl.tblPr
        borders = OxmlElement("w:tblBorders")
        for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
            b = OxmlElement(f"w:{edge}")
            b.set(qn("w:val"), "single")
            b.set(qn("w:sz"), "4")
            b.set(qn("w:color"), "000000")
            borders.append(b)
        tbl_pr.append(borders)
        align = align or ["center"] * len(headers)
        all_rows = [headers] + [list(r) for r in rows]
        for i, values in enumerate(all_rows):
            row = table.rows[i]
            if i == 0:
                trpr = row._tr.get_or_add_trPr()
                _set(trpr, "w:tblHeader")
            for j, value in enumerate(values):
                cell = row.cells[j]
                par = cell.paragraphs[0]
                pf = par.paragraph_format
                pf.first_line_indent = Cm(0)
                pf.line_spacing = 1.0
                pf.space_before = pf.space_after = Pt(1)
                par.alignment = {"center": WD_ALIGN_PARAGRAPH.CENTER,
                                 "left": WD_ALIGN_PARAGRAPH.LEFT,
                                 "right": WD_ALIGN_PARAGRAPH.RIGHT}[align[j] if i else "center"]
                self._runs(par, str(value), size=size,
                           bold=(i == 0) or (bold_last and i == len(all_rows) - 1))
        if widths:
            for row in table.rows:
                for cell, w in zip(row.cells, widths):
                    cell.width = Cm(w)
        spacer = self.doc.add_paragraph()
        spacer.paragraph_format.space_after = Pt(0)
        spacer.paragraph_format.line_spacing = 0.8
        return self.tables

    def code(self, text, size=5):
        """Листинг: Consolas, без подложки; перенесенная часть длинной строки с отступом."""
        for line in text.rstrip("\n").split("\n"):
            par = self.doc.add_paragraph()
            pf = par.paragraph_format
            pf.left_indent = Cm(0.3)
            pf.first_line_indent = Cm(-0.3)
            pf.line_spacing = 1.0
            pf.space_before = pf.space_after = Pt(0)
            par.alignment = WD_ALIGN_PARAGRAPH.LEFT
            run = par.add_run(line if line else " ")
            run.font.name = "Consolas"
            run._element.rPr.rFonts.set(qn("w:hAnsi"), "Consolas")
            run._element.rPr.rFonts.set(qn("w:cs"), "Consolas")
            run.font.size = Pt(size)
            mark = OxmlElement("w:rPr")
            _set(mark, "w:sz", **{"w:val": int(size * 2)})
            par._p.get_or_add_pPr().append(mark)

    def code_title(self, text):
        """Подпись файла внутри листинга."""
        par = self._runs(self.doc.add_paragraph(), f"**{text}**", size=6)
        pf = par.paragraph_format
        pf.first_line_indent = Cm(0)
        pf.line_spacing = 1.0
        pf.space_before, pf.space_after = Pt(4), Pt(2)
        pf.keep_with_next = True

    def _section_break(self, columns=1, continuous=False):
        """Завершает текущий раздел копией основного sectPr (поля, колонтитул с номером)."""
        final = self.doc.element.body.find(qn("w:sectPr"))
        sect = deepcopy(final)
        for old in sect.findall(qn("w:type")):
            sect.remove(old)
        if continuous:
            kind = OxmlElement("w:type")
            kind.set(qn("w:val"), "continuous")
            sect.find(qn("w:pgSz")).addprevious(kind)
        cols = sect.find(qn("w:cols"))
        if cols is None:
            cols = OxmlElement("w:cols")
            sect.find(qn("w:docGrid")).addprevious(cols)
        cols.attrib.clear()
        cols.set(qn("w:space"), "284")
        if columns > 1:
            cols.set(qn("w:num"), str(columns))
        par = OxmlElement("w:p")
        ppr = OxmlElement("w:pPr")
        ppr.append(sect)
        par.append(ppr)
        final.addprevious(par)

    def code_columns(self, files, columns=3):
        """Листинги нескольких файлов в несколько колонок; затем снова одна колонка."""
        self._section_break(columns=1)
        for title, text in files:
            self.code_title(title)
            self.code(text)
        self._section_break(columns=columns, continuous=True)
        final = self.doc.element.body.find(qn("w:sectPr"))
        if final.find(qn("w:type")) is None:
            kind = OxmlElement("w:type")
            kind.set(qn("w:val"), "continuous")
            final.find(qn("w:pgSz")).addprevious(kind)

    def page_break(self):
        from docx.enum.text import WD_BREAK
        self.doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    def structural(self, text):
        par = self.doc.add_paragraph(text, style="Структурный заголовок")
        par.paragraph_format.page_break_before = True
        return par

    def save(self, path):
        self.doc.core_properties.author = STUDENT
        self.doc.core_properties.title = path.stem
        self.doc.save(str(path))


# ---------- картинка вывода консоли ----------

def console_png(lines, path, title=None, width_chars=None, font_size=15):
    """Рисует реальный текст вывода программы в окне в стиле консоли Windows."""
    from PIL import Image, ImageDraw, ImageFont
    font_file = Path("C:/Windows/Fonts/consola.ttf")
    font = ImageFont.truetype(str(font_file), font_size) if font_file.exists() \
        else ImageFont.load_default()
    cw, ch = font.getlength("M"), int(font_size * 1.32)
    wrap = 118
    wrapped = []
    for line in lines:
        while len(line) > wrap:
            wrapped.append(line[:wrap])
            line = "    " + line[wrap:]
        wrapped.append(line)
    lines = wrapped
    width_chars = width_chars or max(len(s) for s in lines)
    pad, bar = 14, (30 if title else 0)
    img = Image.new("RGB", (pad * 2 + int(cw * width_chars) + 2, pad * 2 + bar + ch * len(lines)), "#0C0C0C")
    draw = ImageDraw.Draw(img)
    if title:
        draw.rectangle([0, 0, img.width, bar], fill="#2B2B2B")
        draw.text((pad, 7), title, font=font, fill="#DADADA")
    for k, line in enumerate(lines):
        color = "#F2F2F2"
        if line.startswith("===") or line.isupper():
            color = "#61D6D6"
        elif line.lstrip().startswith(("+", "|")):
            color = "#CCCCCC"
        draw.text((pad, pad + bar + k * ch), line[:width_chars], font=font, fill=color)
    img.save(path)
    return path

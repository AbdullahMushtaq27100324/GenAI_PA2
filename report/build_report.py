"""Build report/PA2_report.docx and report/PA2_report.pdf from report/report.md.

Uses only tools already on the machine: pandoc (Markdown -> DOCX with a restyled reference document),
Python's standard library (table post-processing) and LibreOffice (DOCX -> PDF).

    python3 report/build_report.py
"""

from __future__ import annotations

import io
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC, DOCX, PDF = HERE / "report.md", HERE / "PA2_report.docx", HERE / "PA2_report.pdf"

SERIF, SANS, MONO = "Liberation Serif", "Liberation Sans", "Liberation Mono"
INK, ACCENT, RULE = "1A1A1A", "1F3A5F", "404040"


def fonts(name: str) -> str:
    return f'<w:rFonts w:ascii="{name}" w:hAnsi="{name}" w:eastAsia="{name}" w:cs="{name}"/>'


def pstyle(sid, name, based="Normal", ppr="", rpr="", extra=""):
    return (f'<w:style w:type="paragraph" w:customStyle="1" w:styleId="{sid}"><w:name w:val="{name}"/>'
            f'<w:basedOn w:val="{based}"/>{extra}<w:qFormat/><w:pPr>{ppr}</w:pPr><w:rPr>{rpr}</w:rPr></w:style>')


STYLES = {
    "Normal": '<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/>'
              '<w:pPr><w:spacing w:before="0" w:after="0" w:line="252" w:lineRule="auto"/></w:pPr></w:style>',
    "BodyText": pstyle("BodyText", "Body Text", ppr='<w:spacing w:before="0" w:after="90"/><w:jc w:val="both"/>'),
    "FirstParagraph": pstyle("FirstParagraph", "First Paragraph", "BodyText"),
    "Compact": pstyle("Compact", "Compact", "BodyText", ppr='<w:spacing w:before="0" w:after="30"/><w:jc w:val="left"/>'),
    "SourceCode": pstyle("SourceCode", "Source Code", ppr='<w:spacing w:before="20" w:after="60"/><w:jc w:val="left"/>'
                         '<w:shd w:val="clear" w:color="auto" w:fill="F3F4F6"/>',
                         rpr=fonts(MONO) + '<w:sz w:val="14"/><w:szCs w:val="14"/>'),
    "Title": pstyle("Title", "Title", ppr='<w:spacing w:before="0" w:after="60"/><w:jc w:val="center"/>',
                    rpr=fonts(SANS) + f'<w:b/><w:color w:val="{ACCENT}"/><w:sz w:val="32"/><w:szCs w:val="32"/>'),
    "Subtitle": pstyle("Subtitle", "Subtitle", ppr='<w:spacing w:before="0" w:after="40"/><w:jc w:val="center"/>',
                       rpr=fonts(SANS) + '<w:sz w:val="21"/><w:szCs w:val="21"/>'),
    "Author": pstyle("Author", "Author", ppr='<w:spacing w:before="0" w:after="0"/><w:jc w:val="center"/>',
                     rpr=fonts(SANS) + '<w:sz w:val="18"/><w:szCs w:val="18"/>'),
    "Date": pstyle("Date", "Date", ppr='<w:spacing w:before="0" w:after="80"/><w:jc w:val="center"/>',
                   rpr=fonts(SANS) + '<w:sz w:val="18"/><w:szCs w:val="18"/>'),
    "AbstractTitle": pstyle("AbstractTitle", "Abstract Title", ppr='<w:spacing w:before="40" w:after="20"/><w:jc w:val="center"/>',
                            rpr=fonts(SANS) + '<w:b/><w:sz w:val="18"/><w:szCs w:val="18"/>'),
    "Abstract": pstyle("Abstract", "Abstract", ppr='<w:spacing w:before="0" w:after="120"/><w:ind w:left="454" w:right="454"/><w:jc w:val="both"/>',
                       rpr='<w:sz w:val="19"/><w:szCs w:val="19"/>'),
    "Heading1": pstyle("Heading1", "heading 1", ppr='<w:keepNext/><w:keepLines/><w:spacing w:before="200" w:after="70"/><w:outlineLvl w:val="0"/>'
                       '<w:pBdr><w:bottom w:val="single" w:sz="4" w:space="1" w:color="B0B0B0"/></w:pBdr>',
                       rpr=fonts(SANS) + f'<w:b/><w:color w:val="{ACCENT}"/><w:sz w:val="23"/><w:szCs w:val="23"/>'),
    "Heading2": pstyle("Heading2", "heading 2", ppr='<w:keepNext/><w:keepLines/><w:spacing w:before="120" w:after="40"/><w:outlineLvl w:val="1"/>',
                       rpr=fonts(SANS) + f'<w:b/><w:color w:val="{ACCENT}"/><w:sz w:val="20"/><w:szCs w:val="20"/>'),
    "Caption": pstyle("Caption", "caption", ppr='<w:spacing w:before="40" w:after="120"/><w:jc w:val="both"/>',
                      rpr=fonts(SANS) + '<w:sz w:val="16"/><w:szCs w:val="16"/>'),
    "TableCaption": pstyle("TableCaption", "Table Caption", "Caption", ppr='<w:keepNext/><w:spacing w:before="100" w:after="50"/>'),
    "ImageCaption": pstyle("ImageCaption", "Image Caption", "Caption", ppr='<w:spacing w:before="20" w:after="140"/>'),
    "Figure": pstyle("Figure", "Figure", ppr='<w:spacing w:before="80" w:after="0"/><w:jc w:val="center"/>'),
    "CaptionedFigure": pstyle("CaptionedFigure", "Captioned Figure", "Figure", ppr='<w:keepNext/>'),
    "Reference": pstyle("Reference", "Reference", ppr='<w:spacing w:before="0" w:after="20"/><w:ind w:left="340" w:hanging="340"/>',
                        rpr='<w:sz w:val="16"/><w:szCs w:val="16"/>'),
    "TableText": pstyle("TableText", "Table Text", ppr='<w:spacing w:before="10" w:after="10" w:line="220" w:lineRule="auto"/><w:jc w:val="left"/>',
                        rpr=fonts(SANS) + '<w:sz w:val="14"/><w:szCs w:val="14"/>'),
    "VerbatimChar": '<w:style w:type="character" w:customStyle="1" w:styleId="VerbatimChar"><w:name w:val="Verbatim Char"/>'
                    f'<w:rPr>{fonts(MONO)}<w:sz w:val="17"/></w:rPr></w:style>',
    "Hyperlink": '<w:style w:type="character" w:styleId="Hyperlink"><w:name w:val="Hyperlink"/>'
                 '<w:rPr><w:color w:val="1F5FBF"/></w:rPr></w:style>',
}

FOOTER = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
          '<w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
          'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
          '<w:p><w:pPr><w:jc w:val="center"/></w:pPr>'
          f'<w:r><w:rPr>{fonts(SANS)}<w:sz w:val="16"/></w:rPr><w:fldChar w:fldCharType="begin"/></w:r>'
          f'<w:r><w:rPr>{fonts(SANS)}<w:sz w:val="16"/></w:rPr><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r>'
          f'<w:r><w:rPr>{fonts(SANS)}<w:sz w:val="16"/></w:rPr><w:fldChar w:fldCharType="separate"/></w:r>'
          f'<w:r><w:rPr>{fonts(SANS)}<w:sz w:val="16"/></w:rPr><w:t>1</w:t></w:r>'
          f'<w:r><w:rPr>{fonts(SANS)}<w:sz w:val="16"/></w:rPr><w:fldChar w:fldCharType="end"/></w:r></w:p></w:ftr>')

SECTPR = ('<w:sectPr><w:footerReference w:type="default" r:id="rIdPageFooter"/>'
          '<w:pgSz w:w="11906" w:h="16838"/>'  # A4
          '<w:pgMar w:top="1020" w:right="1020" w:bottom="1020" w:left="1020" w:header="454" w:footer="454" w:gutter="0"/>'
          '</w:sectPr>')


def make_reference(path: Path) -> None:
    default = subprocess.run(["pandoc", "--print-default-data-file", "reference.docx"], capture_output=True, check=True).stdout
    zin = zipfile.ZipFile(io.BytesIO(default))
    out = zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED)
    for item in zin.infolist():
        data = zin.read(item.filename).decode("utf-8") if item.filename.endswith((".xml", ".rels")) else zin.read(item.filename)
        if item.filename == "word/styles.xml":
            data = re.sub(r"<w:rPrDefault>.*?</w:rPrDefault>",
                          f'<w:rPrDefault><w:rPr>{fonts(SERIF)}<w:sz w:val="20"/><w:szCs w:val="20"/>'
                          '<w:lang w:val="en-US"/></w:rPr></w:rPrDefault>', data, flags=re.S)
            data = re.sub(r"<w:pPrDefault>.*?</w:pPrDefault>", "<w:pPrDefault><w:pPr><w:spacing w:after=\"0\"/></w:pPr></w:pPrDefault>",
                          data, flags=re.S)
            for sid, xml in STYLES.items():
                pattern = re.compile(r'<w:style [^>]*w:styleId="%s".*?</w:style>' % sid, re.S)
                data = pattern.sub(lambda _m, x=xml: x, data, count=1) if pattern.search(data) else data.replace("</w:styles>", xml + "</w:styles>")
        elif item.filename == "word/document.xml":
            data = re.sub(r"<w:sectPr\s*/>|<w:sectPr>.*?</w:sectPr>", SECTPR, data, flags=re.S)
        elif item.filename == "word/_rels/document.xml.rels":
            data = data.replace("</Relationships>",
                                '<Relationship Id="rIdPageFooter" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
                                'relationships/footer" Target="footer1.xml"/></Relationships>')
        elif item.filename == "[Content_Types].xml":
            data = data.replace("</Types>", '<Override PartName="/word/footer1.xml" ContentType="application/'
                                'vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/></Types>')
        out.writestr(item, data.encode("utf-8") if isinstance(data, str) else data)
    out.writestr("word/footer1.xml", FOOTER)
    out.close()


SPACER = '<w:p><w:pPr><w:spacing w:before="0" w:after="0" w:line="140" w:lineRule="exact"/></w:pPr></w:p>'


def style_table(tbl: str) -> str:
    """Booktabs-like rules, a shaded bold header row, compact sans text."""
    tbl = tbl.replace('<w:pStyle w:val="Compact" />', '<w:pStyle w:val="TableText" />').replace(
        '<w:pStyle w:val="Compact"/>', '<w:pStyle w:val="TableText"/>')
    borders = (f'<w:tblBorders><w:top w:val="single" w:sz="8" w:space="0" w:color="{RULE}"/>'
               f'<w:bottom w:val="single" w:sz="8" w:space="0" w:color="{RULE}"/>'
               '<w:insideH w:val="single" w:sz="2" w:space="0" w:color="D4D4D0"/></w:tblBorders>'
               '<w:tblCellMar><w:top w:w="15" w:type="dxa"/><w:left w:w="57" w:type="dxa"/>'
               '<w:bottom w:w="15" w:type="dxa"/><w:right w:w="57" w:type="dxa"/></w:tblCellMar>')
    tbl = re.sub(r'<w:rStyle w:val="VerbatimChar"\s*/>', '<w:rStyle w:val="VerbatimChar"/><w:sz w:val="13"/>', tbl)
    tbl = re.sub(r"<w:tblBorders>.*?</w:tblBorders>", "", tbl, flags=re.S)
    tbl = re.sub(r"<w:tblCellMar>.*?</w:tblCellMar>", "", tbl, flags=re.S)
    tbl = tbl.replace("</w:tblPr>", borders + "</w:tblPr>", 1)
    first_tr = re.search(r"<w:tr>.*?</w:tr>|<w:tr .*?</w:tr>", tbl, re.S)
    if first_tr:
        row = first_tr.group(0)
        row = re.sub(r"<w:tcPr>", '<w:tcPr><w:shd w:val="clear" w:color="auto" w:fill="ECEDEF"/>'
                     f'<w:tcBorders><w:bottom w:val="single" w:sz="6" w:space="0" w:color="{RULE}"/></w:tcBorders>', row)
        row = re.sub(r"<w:r>(?!<w:rPr>)", "<w:r><w:rPr><w:b/></w:rPr>", row)
        row = re.sub(r"<w:r><w:rPr>(?!<w:b/>)", "<w:r><w:rPr><w:b/>", row)
        tbl = tbl.replace(first_tr.group(0), row, 1)
    return tbl


def postprocess(path: Path) -> None:
    zin = zipfile.ZipFile(path)
    buf = io.BytesIO()
    zout = zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED)
    for item in zin.infolist():
        data = zin.read(item.filename)
        if item.filename == "word/document.xml":
            xml = data.decode("utf-8")
            xml = re.sub(r"<w:tbl>.*?</w:tbl>", lambda m: style_table(m.group(0)) + SPACER, xml, flags=re.S)
            data = xml.encode("utf-8")
        zout.writestr(item, data)
    zout.close()
    path.write_bytes(buf.getvalue())


def main() -> None:
    ref = HERE / "_reference.docx"
    make_reference(ref)
    subprocess.run(["pandoc", SRC.name, "-o", DOCX.name, "--reference-doc", ref.name, "--resource-path", "."],
                   cwd=HERE, check=True)
    postprocess(DOCX)
    ref.unlink()
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(HERE), str(DOCX)],
                   check=True, capture_output=True)
    pages = subprocess.run(["pdfinfo", str(PDF)], capture_output=True, text=True).stdout
    n = int(re.search(r"Pages:\s+(\d+)", pages).group(1))
    print(f"wrote {DOCX.name} and {PDF.name}: {n} pages")
    if n > 8:
        print("WARNING: report exceeds the 8-page limit", file=sys.stderr)


if __name__ == "__main__":
    if not shutil.which("pandoc") or not shutil.which("soffice"):
        sys.exit("needs pandoc and LibreOffice (soffice)")
    main()

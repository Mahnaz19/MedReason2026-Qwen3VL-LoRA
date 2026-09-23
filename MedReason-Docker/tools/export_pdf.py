#!/usr/bin/env python3
"""Export model description docx to PDF via Word COM (Windows only)."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# wdExportFormatPDF
WD_EXPORT_FORMAT_PDF = 17


def export_pdf(docx: Path, pdf: Path) -> None:
    try:
        import win32com.client  # type: ignore
    except ImportError as exc:
        raise SystemExit("pip install pywin32") from exc

    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    doc = None
    try:
        doc = word.Documents.Open(str(docx.resolve()), ReadOnly=True)
        doc.ExportAsFixedFormat(str(pdf.resolve()), WD_EXPORT_FORMAT_PDF)
    finally:
        if doc is not None:
            doc.Close(False)
        word.Quit()
        time.sleep(1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--docx",
        type=Path,
        default=Path("../data/MedReason26_Model_Description_FILLED.docx"),
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=Path("../data/MedReason26_Model_Description_FILLED.pdf"),
    )
    args = parser.parse_args()
    if not args.docx.is_file():
        raise SystemExit(f"Missing docx: {args.docx}")
    args.pdf.parent.mkdir(parents=True, exist_ok=True)
    export_pdf(args.docx, args.pdf)
    print(f"[export_pdf] wrote {args.pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

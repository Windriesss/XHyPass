#!/usr/bin/env python3
"""Reject Type 3 fonts in generated publication PDFs."""

from __future__ import annotations

import argparse
from pathlib import Path

from pypdf import PdfReader


def _font_records(resources, seen: set[tuple[int, int]]) -> list[tuple[str, str]]:
    if resources is None:
        return []
    resources = resources.get_object()
    records: list[tuple[str, str]] = []
    for name, reference in resources.get("/Font", {}).items():
        identity = (reference.idnum, reference.generation) if hasattr(reference, "idnum") else id(reference)
        if identity in seen:
            continue
        seen.add(identity)
        font = reference.get_object()
        records.append((str(name), str(font.get("/Subtype", "unknown"))))
        for descendant in font.get("/DescendantFonts", []):
            child = descendant.get_object()
            records.append((str(name), str(child.get("/Subtype", "unknown"))))
    for reference in resources.get("/XObject", {}).values():
        xobject = reference.get_object()
        records.extend(_font_records(xobject.get("/Resources"), seen))
    return records


def check_pdf(path: Path) -> list[str]:
    reader = PdfReader(path)
    failures: list[str] = []
    subtypes: set[str] = set()
    for page_number, page in enumerate(reader.pages, 1):
        records = _font_records(page.get("/Resources"), set())
        subtypes.update(subtype for _, subtype in records)
        for name, subtype in records:
            if subtype == "/Type3":
                failures.append(f"page {page_number}: {name} uses /Type3")
    print(f"[FONT] {path}: {', '.join(sorted(subtypes)) or 'no fonts'}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdfs", nargs="+", type=Path)
    args = parser.parse_args()
    failed = False
    for path in args.pdfs:
        if not path.is_file():
            print(f"[FAIL] missing PDF: {path}")
            failed = True
            continue
        failures = check_pdf(path)
        if failures:
            failed = True
            for failure in failures:
                print(f"[FAIL] {path}: {failure}")
    if not failed:
        print(f"Validated {len(args.pdfs)} PDF(s): no Type 3 fonts")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

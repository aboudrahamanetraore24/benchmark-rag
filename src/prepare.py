"""
Lit les fichiers, extrait et nettoie le texte.

Usage :
    python -m src.prepare
"""

import argparse
import hashlib
import json
import logging
import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader
from docx import Document
from pptx import Presentation

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("prepare")

RAW_DIR = Path("data/raw")
EXTRACTED_OUT = Path("data/extracted/documents_extracted.jsonl")
SUPPORTED = {".pdf": "pdf", ".docx": "docx", ".pptx": "pptx", ".txt": "txt"}

MIN_CHARS = 100


# ----------------------------------------------------------------------
# Identifiant stable du document.
# ----------------------------------------------------------------------
def doc_id_for(path_or_str: str) -> str:
    return hashlib.sha1(path_or_str.encode()).hexdigest()[:12]


# ----------------------------------------------------------------------
# Extraction : une fonction par format.
# ----------------------------------------------------------------------
def extract_pdf(path: Path, doc_id: str, source: str) -> dict:
    text = "\n\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    return {"doc_id": doc_id, "source_file": source, "file_type": "pdf", "text": text}


def extract_docx(path: Path, doc_id: str, source: str) -> dict:
    text = "\n\n".join(p.text for p in Document(str(path)).paragraphs)
    return {"doc_id": doc_id, "source_file": source, "file_type": "docx", "text": text}


def extract_pptx(path: Path, doc_id: str, source: str) -> dict:
    slides = ["\n".join(s.text_frame.text for s in slide.shapes if s.has_text_frame)
              for slide in Presentation(str(path)).slides]
    text = "\n\n".join(slides)
    return {"doc_id": doc_id, "source_file": source, "file_type": "pptx", "text": text}


def extract_txt(path: Path, doc_id: str, source: str) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return {"doc_id": doc_id, "source_file": source, "file_type": "txt", "text": text}


def extract_one(path: Path) -> dict:
    rel = path.relative_to(RAW_DIR).as_posix()
    doc_id = doc_id_for(rel)
    file_type = SUPPORTED[path.suffix.lower()]
    if file_type == "pdf":
        return extract_pdf(path, doc_id, rel)
    if file_type == "docx":
        return extract_docx(path, doc_id, rel)
    if file_type == "pptx":
        return extract_pptx(path, doc_id, rel)
    return extract_txt(path, doc_id, rel)


# ----------------------------------------------------------------------
# Nettoyage du texte.
# ----------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"-\n(\w)", r"\1", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ----------------------------------------------------------------------
# Extraction depuis data/raw/.
# ----------------------------------------------------------------------
def run_from_raw():
    files = [p for p in RAW_DIR.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED]
    log.info(f"Fichiers collectés : {len(files)}")

    EXTRACTED_OUT.parent.mkdir(parents=True, exist_ok=True)
    n_clean = 0

    with EXTRACTED_OUT.open("w", encoding="utf-8") as f:
        for path in files:
            try:
                row = extract_one(path)
            except Exception as e:
                log.warning(f"Échec : {path.name} ({e})")
                continue
            row["text"] = clean_text(row["text"])
            if len(row["text"]) >= MIN_CHARS:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_clean += 1
            log.info(f"{path.name} -> extrait")

    log.info(f"Terminé : {len(files)} fichiers | {n_clean} nettoyés")


# ----------------------------------------------------------------------
# Point d'entrée.
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Extraction du corpus vers documents_extracted.jsonl")
    args = parser.parse_args()
    run_from_raw()


if __name__ == "__main__":
    main()

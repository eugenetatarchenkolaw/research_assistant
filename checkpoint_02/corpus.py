"""Versioned PDF ingestion. Search offsets refer to cleaned document text."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
import json
import re
import unicodedata

import fitz
import requests

ROOT = Path(__file__).resolve().parent
TOKEN_RE = re.compile(r"[a-zA-Z\u0410-\u044f\u0401\u04510-9]+(?:[.-][0-9]+)*")
PARSER_VERSION = "pymupdf-clean-v1"


@dataclass(frozen=True)
class Page:
    number: int
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class Document:
    document_id: str
    title: str
    text: str
    pages: tuple[Page, ...]
    pdf_sha256: str
    source_url: str

    @property
    def text_sha256(self) -> str:
        return sha256(self.text.encode()).hexdigest()


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().replace("\u0451", "\u0435").split())


def clean_page(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\u00ad\s*\n\s*", "", text).replace("\u00ad", "")
    text = re.sub(r"(?<=[A-Za-z\u0410-\u044f\u0401\u0451])-\s*\n\s*(?=[a-z\u0430-\u044f\u0451])", "", text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    if lines and re.match(r"\u0413\u041e\u0421\u0422.*(?:19|20)\d{2}$", lines[0].replace(" ", "")):
        lines.pop(0)
    if lines and re.fullmatch(r"(?:\d{1,3}|[IVX]+)", lines[-1]):
        lines.pop()
    return "\n".join(lines).strip()


def load_sources(root: Path = ROOT) -> list[dict]:
    return json.loads((root / "data/sources.json").read_text(encoding="utf-8"))


def ensure_pdfs(root: Path = ROOT) -> list[Path]:
    """Download once, verify bytes before use, never replace a pinned version."""
    folder = root / "data/raw"
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for spec in load_sources(root):
        path = folder / f"{spec['document_id']}.pdf"
        if not path.exists():
            errors = []
            for url in [spec["url"], *spec.get("mirror_urls", [])]:
                try:
                    response = requests.get(url, timeout=(15, 90))
                    response.raise_for_status()
                    content = response.content
                    if not content.startswith(b"%PDF-"):
                        raise ValueError("Server did not return a PDF")
                    if sha256(content).hexdigest() != spec["sha256"]:
                        raise ValueError("Downloaded PDF differs from the pinned SHA-256")
                    temp = path.with_suffix(".tmp")
                    temp.write_bytes(content)
                    temp.replace(path)
                    break
                except (requests.RequestException, ValueError) as exc:
                    errors.append(f"{url}: {type(exc).__name__}: {exc}")
            else:
                raise RuntimeError(f"Could not obtain {path.name}. Upload the pinned PDF to {folder}. " + "; ".join(errors))
        if sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {path}. Do not mix corpus versions.")
        paths.append(path)
    return paths


def load_corpus(root: Path = ROOT, download: bool = True) -> tuple[list[Document], list[dict]]:
    if download:
        ensure_pdfs(root)
    documents, audit = [], []
    for spec in load_sources(root):
        path = root / "data/raw" / f"{spec['document_id']}.pdf"
        actual_sha = sha256(path.read_bytes()).hexdigest()
        if actual_sha != spec["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {path.name}")
        pages, parts, offset = [], [], 0
        with fitz.open(path) as pdf:
            if len(pdf) != spec["pages"]:
                raise ValueError(f"Page count changed: {path.name}")
            for number, page in enumerate(pdf, 1):
                raw = page.get_text(sort=True)
                text = clean_page(raw)
                included = spec["search_pages"][0] <= number <= spec["search_pages"][1]
                tokens = TOKEN_RE.findall(text)
                letters = [c for c in text if c.isalpha()]
                audit.append(dict(document_id=spec["document_id"], page=number, raw_chars=len(raw),
                    clean_chars=len(text), tokens=len(tokens), included=included,
                    replacements=raw.count("\ufffd"), soft_hyphens=raw.count("\u00ad"),
                    latin_share=sum("a" <= c.lower() <= "z" for c in letters) / max(1, len(letters)),
                    digit_share=sum(c.isdigit() for c in text) / max(1, len(text)),
                    sparse=len(tokens) < 50, normalized_sha256=sha256(normalize(text).encode()).hexdigest()))
                if included and text:
                    pages.append(Page(number, offset, offset + len(text), text))
                    parts.append(text)
                    offset += len(text) + 2
        documents.append(Document(spec["document_id"], spec["title"], "\n\n".join(parts),
                                  tuple(pages), actual_sha, spec["url"]))
    if not any(d.text for d in documents):
        raise ValueError("The corpus has no searchable text; OCR is not implemented")
    return documents, audit


def corpus_fingerprint(documents: list[Document]) -> str:
    payload = [(d.document_id, d.pdf_sha256, d.text_sha256) for d in documents]
    return sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

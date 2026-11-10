"""Word-token windows and sentence-aware windows with exact source offsets."""
from __future__ import annotations
from bisect import bisect_right
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import re

from .corpus import Document, TOKEN_RE


@dataclass(frozen=True)
class ChunkConfig:
    strategy: str = "sentence"
    size: int = 256
    overlap: int = 32

    def __post_init__(self):
        if self.strategy not in {"token", "sentence"}:
            raise ValueError("strategy must be token or sentence")
        if not isinstance(self.size, int) or not isinstance(self.overlap, int):
            raise ValueError("size and overlap must be integers")
        if self.size < 1 or not 0 <= self.overlap < self.size:
            raise ValueError("Require size > 0 and 0 <= overlap < size")

    @property
    def key(self) -> str:
        return f"{self.strategy}-{self.size}-{self.overlap}"


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    start: int
    end: int
    text: str
    pages: tuple[int, ...]
    tokens: int


def split_document(document: Document, config: ChunkConfig) -> list[Chunk]:
    matches = list(TOKEN_RE.finditer(document.text))
    if not matches:
        return []
    starts = [m.start() for m in matches]
    sentence_ends = sorted({bisect_right(starts, m.start()) for m in re.finditer(
        r"(?<=[.!?])\s+(?=[A-Z\u0410-\u042f\u04010-9])|\n(?=\d+(?:\.\d+)*\s)", document.text)})
    chunks, begin = [], 0
    while begin < len(matches):
        end = min(begin + config.size, len(matches))
        if config.strategy == "sentence" and end < len(matches):
            position = bisect_right(sentence_ends, end) - 1
            if position >= 0 and sentence_ends[position] >= begin + max(config.overlap + 1, config.size // 2):
                end = sentence_ends[position]
        char_start, char_end = matches[begin].start(), matches[end - 1].end()
        identity = f"{document.document_id}:{document.text_sha256}:{config.key}:{char_start}:{char_end}"
        pages = tuple(p.number for p in document.pages if p.start < char_end and p.end > char_start)
        chunks.append(Chunk(sha256(identity.encode()).hexdigest()[:20], document.document_id,
                            char_start, char_end, document.text[char_start:char_end], pages, end - begin))
        if end == len(matches):
            break
        begin = end - config.overlap
    return chunks


def split_corpus(documents: list[Document], config: ChunkConfig) -> list[Chunk]:
    chunks = [c for d in documents for c in split_document(d, config)]
    if len({c.chunk_id for c in chunks}) != len(chunks):
        raise ValueError("Chunk IDs are not unique; document IDs must be unique")
    return chunks

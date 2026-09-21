from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class StructuralChunk:
    document_id: str
    content_hash: str
    page_start: int | None
    page_end: int | None
    section_path: str
    section_type: str
    section_title: str
    section_order: int
    is_abstract: bool
    is_reference: bool
    block_type: str
    chunk_index: int
    text: str
    ingestion_generation: str
    parser_version: str
    embedding_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def structural_chunks(
    *,
    document_id: str,
    content_hash: str,
    markdown: str,
    generation: str,
    parser_version: str,
    embedding_version: str,
    max_chars: int = 1600,
    page_blocks: list[dict[str, Any]] | None = None,
) -> list[StructuralChunk]:
    sections: list[tuple[str, str]] = []
    heading = ""
    buffer: list[str] = []
    for line in markdown.splitlines():
        if re.match(r"^#{1,6}\s+", line):
            if buffer:
                sections.append((heading, "\n".join(buffer).strip()))
                buffer = []
            heading = re.sub(r"^#{1,6}\s+", "", line).strip()
        else:
            buffer.append(line)
    if buffer:
        sections.append((heading, "\n".join(buffer).strip()))
    chunks: list[StructuralChunk] = []
    for section_path, text in sections:
        paragraphs = [item.strip() for item in re.split(r"\n\s*\n", text) if item.strip()]
        current = ""
        for paragraph in paragraphs:
            for part in _bounded_parts(paragraph, max_chars):
                if current and len(current) + len(part) + 2 > max_chars:
                    chunks.append(_chunk(
                        document_id, content_hash, section_path, current,
                        len(chunks), generation, parser_version, embedding_version,
                        page_blocks,
                    ))
                    current = ""
                current = part if not current else current + "\n\n" + part
        if current:
            chunks.append(_chunk(
                document_id, content_hash, section_path, current,
                len(chunks), generation, parser_version, embedding_version,
                page_blocks,
            ))
    return chunks


def _chunk(document_id, content_hash, section_path, text, index, generation,
           parser_version, embedding_version, page_blocks=None):
    pages = _explicit_pages(text, page_blocks or [])
    return StructuralChunk(
        document_id=document_id,
        content_hash=content_hash,
        page_start=min(pages) if pages else None,
        page_end=max(pages) if pages else None,
        section_path=section_path,
        section_type=_section_type(section_path),
        section_title=section_path,
        section_order=index,
        is_abstract=_section_type(section_path) == "abstract",
        is_reference=_section_type(section_path) == "references",
        block_type="section",
        chunk_index=index,
        text=text,
        ingestion_generation=generation,
        parser_version=parser_version,
        embedding_version=embedding_version,
    )


def _section_type(title: str) -> str:
    normalized = re.sub(r"\s+", " ", str(title)).strip().casefold()
    rules = (
        (r"摘要|abstract", "abstract"),
        (r"引言|绪论|introduction|background", "introduction"),
        (r"方法|模型|算法|method|methodology|approach", "method"),
        (r"实验|experiment|evaluation|implementation", "experiment"),
        (r"结果|讨论|result|discussion", "result"),
        (r"结论|总结|conclusion", "conclusion"),
        (r"参考文献|references|bibliography", "references"),
    )
    for pattern, value in rules:
        if re.search(pattern, normalized, re.I):
            return value
    return "body"


def _explicit_pages(text: str, blocks: list[dict[str, Any]]) -> list[int]:
    """Match explicit parser blocks conservatively; unmatched text stays page-less."""
    normalize = lambda value: re.sub(r"\W+", "", str(value), flags=re.UNICODE).casefold()
    chunk_text = normalize(text)
    if not chunk_text:
        return []
    pages: set[int] = set()
    for block in blocks:
        block_text = normalize(block.get("text", ""))
        if len(block_text) < 12:
            continue
        probe = block_text[: min(80, len(block_text))]
        reverse_probe = chunk_text[: min(80, len(chunk_text))]
        if probe not in chunk_text and reverse_probe not in block_text:
            continue
        start, end = block.get("page_start"), block.get("page_end")
        if isinstance(start, int) and isinstance(end, int) and 0 < start <= end:
            pages.update(range(start, end + 1))
    return sorted(pages)


def _bounded_parts(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    sentences = [item.strip() for item in re.split(r"(?<=[。！？.!?；;])\s*", text) if item.strip()]
    result: list[str] = []
    current = ""
    for sentence in sentences or [text]:
        pieces = [sentence[index:index + max_chars] for index in range(0, len(sentence), max_chars)]
        for piece in pieces:
            if current and len(current) + len(piece) > max_chars:
                result.append(current)
                current = ""
            current += piece
            if len(current) == max_chars:
                result.append(current)
                current = ""
    if current:
        result.append(current)
    return result

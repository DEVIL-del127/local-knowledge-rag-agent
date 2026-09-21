from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class DocumentQualityReport:
    status: str
    score: float
    page_count: int
    empty_page_ratio: float
    replacement_character_ratio: float
    structure_blocks: int
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_document(parsed: dict[str, Any]) -> DocumentQualityReport:
    document = parsed.get("document") or {}
    pages = document.get("pages") or parsed.get("pages") or []
    page_count = len(pages) or int(document.get("num_pages") or 0)
    markdown = str(parsed.get("markdown") or parsed.get("text") or "")
    replacement_ratio = markdown.count("\ufffd") / max(1, len(markdown))
    empty_pages = 0
    for page in pages:
        text = str(page.get("text") or page.get("content") or "") if isinstance(page, dict) else str(page)
        if not text.strip():
            empty_pages += 1
    empty_ratio = empty_pages / max(1, page_count)
    blocks = document.get("texts") or document.get("body") or []
    structure_blocks = len(blocks) if isinstance(blocks, list) else 0
    reasons = []
    score = 1.0
    if not markdown.strip():
        score -= 0.7
        reasons.append("no_text")
    score -= min(0.5, empty_ratio * 0.5)
    if empty_ratio > 0.1:
        reasons.append("too_many_empty_pages")
    score -= min(0.4, replacement_ratio * 20)
    if replacement_ratio > 0.01:
        reasons.append("encoding_corruption")
    if structure_blocks == 0:
        score -= 0.1
        reasons.append("no_structural_blocks")
    score = max(0.0, round(score, 3))
    status = "pass" if score >= 0.8 else "review" if score >= 0.55 else "fail"
    return DocumentQualityReport(
        status=status,
        score=score,
        page_count=page_count,
        empty_page_ratio=round(empty_ratio, 4),
        replacement_character_ratio=round(replacement_ratio, 6),
        structure_blocks=structure_blocks,
        reasons=reasons,
    )

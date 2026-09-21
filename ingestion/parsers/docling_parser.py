from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


class ParserUnavailable(RuntimeError):
    pass


class DoclingParser:
    """Optional Docling adapter; importing this module does not import Docling."""

    parser_name = "docling"

    def __init__(self, converter: Any | None = None,
                 cache_dir: str | Path | None = None) -> None:
        self._converter = converter
        self._cache_dir = Path(cache_dir) if cache_dir else None

    def _get_converter(self):
        if self._converter is not None:
            return self._converter
        try:
            from docling.document_converter import DocumentConverter
        except ImportError as exc:
            raise ParserUnavailable(
                "Docling is not installed; install the locked ingestion extra"
            ) from exc
        self._converter = DocumentConverter()
        return self._converter

    def convert(self, pdf_path: str | Path) -> dict[str, Any]:
        path = Path(pdf_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        raw = path.read_bytes()
        source_hash = hashlib.sha256(raw).hexdigest()
        parser_version = _docling_version()
        cache_path = None
        if self._cache_dir is not None:
            cache_key = hashlib.sha256(
                f"docling-cache-v1\0{parser_version}\0{source_hash}".encode()
            ).hexdigest()
            cache_path = self._cache_dir / cache_key[:2] / f"{cache_key}.json"
            if cache_path.is_file():
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if (cached.get("source_hash") == source_hash
                        and cached.get("parser_version") == parser_version):
                    return cached
        result = self._get_converter().convert(str(path))
        document = result.document
        markdown = document.export_to_markdown()
        payload = document.export_to_dict()
        converted = {
            "filename": path.name,
            "source_hash": source_hash,
            "parser": self.parser_name,
            "parser_version": parser_version,
            "markdown": markdown,
            "document": payload,
            "page_blocks": _page_blocks(payload),
        }
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(
                prefix=cache_path.name + ".", suffix=".tmp", dir=cache_path.parent
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(converted, handle, ensure_ascii=False)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, cache_path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return converted


def _page_blocks(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract only explicit Docling page provenance; never infer a page."""
    blocks: list[dict[str, Any]] = []
    for collection_name in ("texts", "tables", "pictures"):
        for item in payload.get(collection_name) or []:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or item.get("orig") or "").strip()
            pages = sorted({
                int(prov.get("page_no")) for prov in item.get("prov") or []
                if isinstance(prov, dict) and isinstance(prov.get("page_no"), int)
                and int(prov["page_no"]) > 0
            })
            if text and pages:
                blocks.append({"text": text, "page_start": pages[0], "page_end": pages[-1]})
    return blocks


def _docling_version() -> str:
    try:
        from importlib.metadata import version
        return version("docling")
    except Exception:
        return "unknown"

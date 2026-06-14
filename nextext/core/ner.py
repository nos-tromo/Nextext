"""Named-entity-recognition agent: HTTP client for the out-of-process ``/gliner`` service.

Nextext no longer hosts GLiNER in-process. NER runs against an HTTP ``/gliner``
endpoint (e.g. ``nos-tromo/vllm-service`` behind the LiteLLM router) that accepts
text plus a label set and returns scored entities. This module owns the wire call
and the client-side tally onto the ``[Category, Entity, Frequency]`` table that
the pipeline, CLI exporter, and API schemas already consume.

The endpoint is located via ``NER_API_BASE``; when it is unset NER is disabled and
callers receive an empty table (see :func:`nextext.utils.env_cfg.load_ner_env`).
Failures are logged and swallowed: a transcript without entities is preferable to
a failed job.
"""

import re
from collections import Counter

import httpx as httpx  # explicit re-export so tests can monkeypatch ner.httpx
import pandas as pd
from loguru import logger

from nextext.utils.env_cfg import load_ner_env

__all__ = ["extract_entities"]

_NER_LABELS = [
    "date",
    "event",
    "fac",
    "group",
    "loc",
    "money",
    "org",
    "person",
    "time",
]
_NER_THRESHOLD = 0.3
_NER_WORD_BUDGET = 512
_SENTENCE_RE = re.compile(
    r".+?(?:[.!?]+[\"')\]]*(?=\s+|$)|\n{2,}|$)",
    re.DOTALL,
)


def _chunk_text(text: str, word_budget: int = _NER_WORD_BUDGET) -> list[str]:
    """Split text into sentence-packed chunks within a word budget.

    Args:
        text (str): Raw input text.
        word_budget (int): Maximum whitespace-delimited words per chunk.

    Returns:
        list[str]: Ordered list of text chunks suitable for repeated NER inference.
    """
    sentences = [m.group(0).strip() for m in _SENTENCE_RE.finditer(text.strip()) if m.group(0).strip()] or [
        text.strip()
    ]
    chunks: list[str] = []
    current_words: list[str] = []
    for sentence in sentences:
        words = sentence.split()
        if len(current_words) + len(words) > word_budget:
            if current_words:
                chunks.append(" ".join(current_words))
            current_words = words[:word_budget]
        else:
            current_words.extend(words)
    if current_words:
        chunks.append(" ".join(current_words))
    return chunks


def extract_entities(text: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Tally named entities for ``text`` via the out-of-process ``/gliner`` service.

    The service URL is ``{NER_API_BASE}/gliner``. When ``NER_API_BASE`` is unset
    NER is disabled: a warning is logged and an empty table is returned so callers
    proceed without entities. Each 512-word chunk is sent as a separate request;
    per-chunk transport/HTTP/JSON errors are logged and skipped so one bad chunk
    never loses the whole transcript's entities. Entities are kept when their
    score is at least :data:`_NER_THRESHOLD` and their text is at least 3
    characters long; labels are upper-cased and counts tallied.

    Args:
        text (str): The transcript text to analyse.
        columns (list[str] | None): Output column names. Defaults to
            ``["Category", "Entity", "Frequency"]``.

    Returns:
        pd.DataFrame: A ``[Category, Entity, Frequency]`` DataFrame. Empty (with
            those columns) when NER is disabled, the text is blank, or every
            request fails.
    """
    if columns is None:
        columns = ["Category", "Entity", "Frequency"]
    empty = pd.DataFrame(columns=pd.Index(columns)).reset_index(drop=True)

    config = load_ner_env()
    if not config.api_base:
        logger.warning(
            "NER requested but NER_API_BASE is unset; returning no entities. "
            "Set NER_API_BASE to enable named-entity recognition."
        )
        return empty
    if not text or not text.strip():
        return empty

    headers: dict[str, str] = {}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    url = f"{config.api_base}/gliner"
    all_entities: list[tuple[str, str]] = []
    for chunk in _chunk_text(text):
        try:
            response = httpx.post(
                url,
                json={"text": chunk, "labels": _NER_LABELS},
                headers=headers,
                timeout=config.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "NER request to {} failed ({}): {}",
                url,
                exc.response.status_code,
                exc.response.text[:500],
            )
            continue
        except (httpx.HTTPError, ValueError, OSError) as exc:
            logger.error("NER request to {} failed: {}", url, exc)
            continue

        if not isinstance(payload, dict):
            logger.error("NER response from {} was not a JSON object; ignoring.", url)
            continue

        for entity in payload.get("entities", []):
            text_val = str(entity.get("text", "")).strip()
            label = str(entity.get("label", "")).strip()
            try:
                score = float(entity.get("score", 0.0))
            except (ValueError, TypeError):
                score = 0.0
            if text_val and label and len(text_val) >= 3 and score >= _NER_THRESHOLD:
                all_entities.append((label.upper(), text_val))

    if not all_entities:
        return empty

    entity_counts = Counter(all_entities)
    return pd.DataFrame(
        [(label, entity, count) for (label, entity), count in entity_counts.items()],
        columns=pd.Index(columns),
    ).reset_index(drop=True)

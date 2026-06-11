"""HTTP client for the remote GLiNER NER service.

This replaces the in-process GLiNER runtime that Nextext previously shipped.
Production deploys reach the service through the central inference router
(LiteLLM ``/gliner`` pass-through, e.g. ``http://vllm-router:4000/gliner``,
Bearer auth); the ner-only deployment shape reaches it directly (e.g.
``http://gliner-only:8000/gliner``, no auth, trust inference-net). The choice
is operator-driven via env vars (see
:func:`nextext.utils.env_cfg.load_ner_client_env`); this module is
mode-agnostic.

Request/response contract (same as docint's client):
``POST {NER_API_BASE}/gliner`` with ``{"text", "labels", "threshold"}`` →
``{"entities": [{"text", "label", "score"}, ...]}``.

Unlike docint, the extractor returns a plain entity list — Nextext has no
relations concept.
"""

from collections.abc import Callable
from typing import Any

import httpx
from loguru import logger

from nextext.utils.env_cfg import NERClientConfig, load_ner_client_env

# Candidate entity labels sent with every request. The remote model is
# zero-shot, so these fully define what gets extracted.
DEFAULT_NER_LABELS: list[str] = [
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


def _build_client(cfg: NERClientConfig) -> httpx.Client:
    """Construct the shared ``httpx.Client`` used for NER calls.

    Args:
        cfg (NERClientConfig): Resolved client configuration.

    Returns:
        httpx.Client: A client bound to the NER base URL, carrying a Bearer
        header when an API key is configured.
    """
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    return httpx.Client(
        base_url=cfg.api_base,
        timeout=cfg.timeout,
        headers=headers,
    )


def build_remote_ner_extractor(
    labels: list[str] | None = None,
    cfg: NERClientConfig | None = None,
) -> Callable[[str], list[dict[str, Any]]]:
    """Create an NER extractor that calls the remote GLiNER endpoint.

    Args:
        labels (list[str] | None): Candidate entity labels to extract. Falls
            back to :data:`DEFAULT_NER_LABELS` when ``None`` or empty.
        cfg (NERClientConfig | None): Override client configuration. When
            ``None``, reads from the environment via
            :func:`nextext.utils.env_cfg.load_ner_client_env`.

    Returns:
        Callable[[str], list[dict[str, Any]]]: A function that takes raw text
        and returns a list of entities ``{"text", "type", "score"}``. On any
        error (network, timeout, non-2xx response, malformed payload) the
        function logs a warning and returns ``[]`` — NER degrades softly
        instead of failing the word-level stage.
    """
    effective_labels = list(labels) if labels else list(DEFAULT_NER_LABELS)
    effective_cfg = cfg if cfg is not None else load_ner_client_env()
    client = _build_client(effective_cfg)
    logger.info(
        "Remote NER extractor ready: api_base={} auth={} threshold={}",
        effective_cfg.api_base,
        "bearer" if effective_cfg.api_key else "none",
        effective_cfg.threshold,
    )

    def _extract(text: str) -> list[dict[str, Any]]:
        """Run remote NER over ``text``.

        Args:
            text (str): Raw text to extract entities from.

        Returns:
            list[dict[str, Any]]: Extracted entities, or ``[]`` on any
            failure (fail-soft).
        """
        if not text.strip():
            return []

        try:
            response = client.post(
                "/gliner",
                json={
                    "text": text,
                    "labels": effective_labels,
                    "threshold": effective_cfg.threshold,
                },
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.warning("Remote NER call failed: {}", exc)
            return []

        raw_entities = payload.get("entities") if isinstance(payload, dict) else None
        if not isinstance(raw_entities, list):
            return []

        entities: list[dict[str, Any]] = []
        for item in raw_entities:
            if not isinstance(item, dict):
                continue
            entity_text = item.get("text")
            entity_label = item.get("label")
            if not entity_text or not entity_label:
                continue
            entities.append(
                {
                    "text": entity_text,
                    "type": entity_label,
                    "score": item.get("score"),
                }
            )

        return entities

    return _extract

"""Tests for ``GET /api/v1/languages``."""

from __future__ import annotations

from fastapi.testclient import TestClient

from nextext.utils.mappings_loader import load_mappings


def test_languages_returns_sorted_whisper_and_target_lists(
    api_client: TestClient,
) -> None:
    """The endpoint should surface both mappings sorted by display name."""
    response = api_client.get("/api/v1/languages")

    assert response.status_code == 200
    body = response.json()
    assert {"whisper", "target"} == set(body.keys())

    whisper_mapping = load_mappings("whisper_languages.json")
    target_mapping = load_mappings("translategemma_languages.json")

    assert len(body["whisper"]) == len(whisper_mapping)
    assert len(body["target"]) == len(target_mapping)

    whisper_names = [entry["name"] for entry in body["whisper"]]
    target_names = [entry["name"] for entry in body["target"]]
    assert whisper_names == sorted(whisper_names)
    assert target_names == sorted(target_names)

    whisper_codes = {entry["code"] for entry in body["whisper"]}
    assert "en" in whisper_codes
    target_codes = {entry["code"] for entry in body["target"]}
    assert "de-DE" in target_codes

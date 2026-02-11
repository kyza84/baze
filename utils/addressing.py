"""Address normalization helpers."""

from __future__ import annotations


def normalize_address(value: str | None) -> str:
    """Normalize on-chain address keys for internal maps/dedup."""
    return str(value or "").strip().lower()


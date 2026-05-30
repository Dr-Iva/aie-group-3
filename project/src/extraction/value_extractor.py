"""Heuristic extraction of numeric values, ranges and units from text."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Set, Tuple

NUMBER_RE = r"[-+]?\d+(?:[.,]\d+)?"
UNIT_PATTERN = r"(?:mV|V|mA|A|uA|µA|Hz|kHz|MHz|GHz|°C|%)"
RANGE_SEPARATORS = r"(?:to|–|—|-|…|through|до)"
VALUE_RE = re.compile(
    rf"(?P<first>{NUMBER_RE})\s*(?:(?P<sep>{RANGE_SEPARATORS})\s*(?P<second>{NUMBER_RE}))?\s*(?P<unit>{UNIT_PATTERN})",
    re.IGNORECASE,
)

REVERSE_UNIT_RE = re.compile(
    rf"(?P<unit>{UNIT_PATTERN})\s*(?P<first>{NUMBER_RE})(?:\s*(?P<sep>{RANGE_SEPARATORS})\s*(?P<second>{NUMBER_RE}))?",
    re.IGNORECASE,
)

WORD_RE = re.compile(r"[A-Za-zА-Яа-я0-9%°]+", re.UNICODE)


@dataclass(slots=True)
class ExtractedValue:
    """Structured extracted value."""

    value: str
    unit: str
    start: int
    end: int
    confidence: float


class ValueExtractor:
    """Best-effort extraction of datasheet values from candidate text."""

    UNIT_CATEGORY_MAP = {
        "voltage": {"v", "mv"},
        "current": {"a", "ma", "ua"},
        "frequency": {"hz", "khz", "mhz", "ghz"},
        "temperature": {"°c"},
        "percent": {"%"},
    }

    KEYWORD_CATEGORY_MAP = {
        "voltage": (
            "voltage",
            "volt",
            "напряж",
            "питания",
            "supply",
            "vcc",
            "vdd",
        ),
        "current": (
            "current",
            "ток",
            "потреблен",
            "consum",
            "sink",
            "source",
        ),
        "frequency": (
            "frequency",
            "частот",
            "clock",
            "oscillat",
            "freq",
        ),
        "temperature": (
            "temperature",
            "температур",
            "thermal",
            "ambient",
        ),
        "percent": (
            "percent",
            "%",
            "duty",
            "ratio",
            "процент",
        ),
    }

    def extract(self, query: str, text: str) -> ExtractedValue | None:
        """Extract the most relevant value for the query from the text."""
        cleaned_text = self._normalize_text(text)
        if not cleaned_text:
            return None

        candidates = self._find_candidates(cleaned_text)
        if not candidates:
            return None

        ranked = []
        for candidate in candidates:
            score = self._score_candidate(query=query, text=cleaned_text, candidate=candidate)
            ranked.append((score, candidate))

        ranked.sort(key=lambda item: item[0], reverse=True)
        best_score, best_candidate = ranked[0]
        return ExtractedValue(
            value=best_candidate["normalized_value"],
            unit=best_candidate["unit"],
            start=best_candidate["start"],
            end=best_candidate["end"],
            confidence=max(0.0, min(1.0, best_score)),
        )

    def _find_candidates(self, text: str) -> List[dict]:
        candidates: List[dict] = []

        for match in VALUE_RE.finditer(text):
            candidates.append(self._candidate_from_match(match, text))

        for match in REVERSE_UNIT_RE.finditer(text):
            candidates.append(self._candidate_from_reverse_match(match, text))

        deduped: List[dict] = []
        seen: set[tuple[int, int, str]] = set()
        for candidate in candidates:
            key = (candidate["start"], candidate["end"], candidate["normalized_value"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)

        return deduped

    def _candidate_from_match(self, match: re.Match[str], text: str) -> dict:
        first = self._normalize_number(match.group("first"))
        second = self._normalize_number(match.group("second"))
        unit = self._normalize_unit(match.group("unit"))

        if second:
            normalized_value = f"{first} to {second} {unit}"
        else:
            normalized_value = f"{first} {unit}"

        return {
            "start": match.start(),
            "end": match.end(),
            "unit": unit,
            "normalized_value": normalized_value.replace("  ", " ").strip(),
            "first": first,
            "second": second,
        }

    def _candidate_from_reverse_match(self, match: re.Match[str], text: str) -> dict:
        first = self._normalize_number(match.group("first"))
        second = self._normalize_number(match.group("second"))
        unit = self._normalize_unit(match.group("unit"))

        if second:
            normalized_value = f"{first} to {second} {unit}"
        else:
            normalized_value = f"{first} {unit}"

        return {
            "start": match.start(),
            "end": match.end(),
            "unit": unit,
            "normalized_value": normalized_value.replace("  ", " ").strip(),
            "first": first,
            "second": second,
        }

    def _score_candidate(self, query: str, text: str, candidate: dict) -> float:
        query_lower = self._normalize_text(query).lower()
        unit = candidate["unit"]
        unit_category = self._unit_category(unit)
        target_category = self._query_category(query_lower)

        score = 0.0

        if target_category and unit_category == target_category:
            score += 3.0
        elif target_category and unit_category is not None:
            score += 1.0

        window = self._window(text, candidate["start"], candidate["end"], radius=140)
        query_tokens = self._query_keywords(query_lower)
        for token in query_tokens:
            if token and token in window:
                score += 0.4

        if self._contains_any(query_lower, ("max", "maximum", "максим", "верх", "upper")):
            score += self._max_bias(candidate)
        elif self._contains_any(query_lower, ("min", "minimum", "миним", "lower")):
            score += self._min_bias(candidate)

        distance = self._distance_to_keywords(text, candidate["start"], query_tokens)
        if distance is not None:
            score += max(0.0, 1.0 - distance / 1000.0)

        if candidate["second"] is not None:
            score += 0.6

        return score

    def _max_bias(self, candidate: dict) -> float:
        values = [self._parse_float(candidate["first"])]
        if candidate["second"] is not None:
            values.append(self._parse_float(candidate["second"]))
        return max(values) / 100.0 if values else 0.0

    def _min_bias(self, candidate: dict) -> float:
        values = [self._parse_float(candidate["first"])]
        if candidate["second"] is not None:
            values.append(self._parse_float(candidate["second"]))
        return 1.0 / (1.0 + min(values)) if values else 0.0

    def _query_keywords(self, query: str) -> List[str]:
        tokens = [token for token in WORD_RE.findall(query.lower()) if len(token) > 1]
        return tokens

    def _query_category(self, query: str) -> str | None:
        for category, keywords in self.KEYWORD_CATEGORY_MAP.items():
            if any(keyword in query for keyword in keywords):
                return category
        return None

    def _unit_category(self, unit: str) -> str | None:
        normalized = unit.lower().replace("µ", "u")
        for category, units in self.UNIT_CATEGORY_MAP.items():
            if normalized in units:
                return category
        return None

    def _distance_to_keywords(self, text: str, candidate_start: int, keywords: Sequence[str]) -> float | None:
        positions = []
        lower_text = text.lower()
        for keyword in keywords:
            idx = lower_text.find(keyword)
            if idx >= 0:
                positions.append(abs(candidate_start - idx))
        if not positions:
            return None
        return float(min(positions))

    @staticmethod
    def _window(text: str, start: int, end: int, radius: int = 120) -> str:
        left = max(0, start - radius)
        right = min(len(text), end + radius)
        return text[left:right].lower()

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = unicodedata.normalize("NFKC", text or "")
        text = text.replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _normalize_number(value: str | None) -> str:
        if value is None:
            return ""
        return value.replace(",", ".").strip()

    @staticmethod
    def _normalize_unit(value: str) -> str:
        normalized = value.replace("µ", "u").strip()
        return normalized

    @staticmethod
    def _parse_float(value: str) -> float:
        try:
            return float(value.replace(",", "."))
        except ValueError:
            return 0.0

    @staticmethod
    def _contains_any(text: str, tokens: Sequence[str]) -> bool:
        return any(token in text for token in tokens)

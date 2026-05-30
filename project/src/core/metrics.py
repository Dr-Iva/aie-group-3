"""Prometheus metrics."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

SEARCH_REQUESTS_TOTAL = Counter(
    "search_requests_total",
    "Total number of search requests",
)

SEARCH_LATENCY_SECONDS = Histogram(
    "search_latency_seconds",
    "Search latency in seconds",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
)

INDEX_SIZE = Gauge(
    "index_size",
    "Number of indexed chunks",
)

"""Prometheus metrics endpoint for the CCTV processor."""

from __future__ import annotations

import logging
import time
from typing import Any

from prometheus_client import CollectorRegistry, start_http_server
from prometheus_client.core import GaugeMetricFamily

logger = logging.getLogger(__name__)


class ProcessorStatsCollector:
    """Expose the processor's live statistics without duplicating counters."""

    _FIELDS = {
        "frames_processed": "cctv_processor_frames_processed_total",
        "frames_dropped": "cctv_processor_frames_dropped_total",
        "events_detected": "cctv_processor_events_detected_total",
        "events_sent": "cctv_processor_events_sent_total",
        "events_filtered": "cctv_processor_events_filtered_total",
        "events_failed": "cctv_processor_events_failed_total",
        "fps": "cctv_processor_fps",
        "avg_inference_ms": "cctv_processor_avg_inference_ms",
        "yolo_postprocess_avg_ms": "cctv_processor_yolo_postprocess_avg_ms",
        "yolo_postprocess_max_ms": "cctv_processor_yolo_postprocess_max_ms",
        "cameras": "cctv_processor_cameras",
    }

    def __init__(self, processor: Any) -> None:
        self._processor = processor
        self._last_frames = 0.0
        self._last_scrape_at = time.monotonic()

    def collect(self):
        try:
            stats = self._processor.get_stats()
        except Exception as exc:
            logger.debug("프로세서 metrics 수집 실패: %s", exc)
            return
        now = time.monotonic()
        frames = float(stats.get("frames_processed", 0) or 0)
        elapsed = now - self._last_scrape_at
        scrape_fps = (frames - self._last_frames) / elapsed if elapsed > 0 else 0.0
        self._last_frames = frames
        self._last_scrape_at = now
        stats["fps"] = max(0.0, scrape_fps)
        stats["avg_inference_ms"] = stats.get("yolo_postprocess_avg_ms", 0.0)
        stats["cameras"] = stats.get("camera_count", 0)
        for field_name, metric_name in self._FIELDS.items():
            value = stats.get(field_name)
            if isinstance(value, (int, float)):
                family = GaugeMetricFamily(metric_name, f"CCTV processor {field_name}")
                family.add_metric([], float(value))
                yield family


def start_processor_metrics_server(processor: Any, port: int) -> Any:
    """Start a background Prometheus HTTP server and return its server handle."""
    registry = CollectorRegistry(auto_describe=True)
    registry.register(ProcessorStatsCollector(processor))
    server, thread = start_http_server(port, registry=registry)
    logger.info("CCTV processor Prometheus metrics server started: port=%s", port)
    return server, thread

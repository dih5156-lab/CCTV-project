"""CCTV Processor의 Prometheus 메트릭 endpoint."""

from __future__ import annotations

import logging
import time
from typing import Any

from prometheus_client import CollectorRegistry, start_http_server
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily

logger = logging.getLogger(__name__)


class ProcessorStatsCollector:
    """카운터를 중복 생성하지 않고 Processor의 현재 통계를 노출한다."""

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
        "primary_metrics_errors": "cctv_processor_primary_metrics_errors_total",
        "model_metrics_errors": "cctv_processor_model_metrics_errors_total",
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

        for field in (
            "source_error_identified",
            "source_error_unidentified",
            "source_eos_identified",
            "source_eos_unidentified",
            "source_unidentified_stop",
            "source_stream_reset_events",
            "source_stream_reset_unidentified",
            "preview_identity_rejected",
            "preview_frames_stored",
            "pipeline_restart_requested",
            "pipeline_restart_rejected",
            "pipeline_restart_executed",
            "pipeline_restart_failed",
            "pipeline_restart_seconds",
        ):
            value = stats.get(field)
            if isinstance(value, (int, float)):
                family = CounterMetricFamily(
                    f"cctv_processor_{field}",
                    f"DeepStream {field}; lifecycle bookkeeping, not frame recovery",
                )
                family.add_metric([], value)
                yield family

        # 전체 프레임 드롭과 별개인 후처리 누락을 노출한다. 매칭 성공은 추론 완료가 아니다.
        context_fields = {
            "context_identity_rejected": "Rejected context identities before post-processing",
            "context_match_expired": "Pending context metadata expired before matching a preview",
            "context_match_evicted": "Pending context metadata evicted at the matching capacity limit",
            "context_match_missed": "Preview frames without matching pending context metadata",
            # 메타 전체 만료/제거 수와 실제 CPU 도착의 누락 사유를 혼합하지 않는다.
            "context_match_missed_no_target": "Unmatched previews after metadata completed without an eligible context target",
            "context_match_missed_expired": "Unmatched previews whose pending context metadata expired",
            "context_match_missed_evicted": "Unmatched previews whose pending context metadata was evicted",
            "context_match_missed_unknown": "Unmatched previews without retained conclusive metadata history",
            "context_frames_matched": "Matched context tasks enqueued; not completed model inference",
            "context_queue_dropped": "Matched context tasks discarded because the worker queue was full",
            # 합의된 주기 조절과 실제 큐 누락·추론 실패를 서로 다른 지표로 보여 준다.
            "context_frames_sampled": "Matched frames skipped by the per-person analysis interval",
            "context_people_sampled": "Repeated person observations skipped by the analysis interval",
            "context_people_busy": "Repeated person observations skipped while that track has unfinished work",
            "context_tasks_completed": "Context tasks whose face and appearance processing completed",
            "context_tasks_failed": "Context tasks that raised during post-processing",
        }
        for field, help_text in context_fields.items():
            value = stats.get(field)
            if isinstance(value, (int, float)):
                family = CounterMetricFamily(f"cctv_processor_{field}", help_text)
                family.add_metric([], value)
                yield family

        yield from self._collect_model_metrics(stats.get("model_inference_stats", {}))
        yield from self._collect_falldata_inline_metrics(
            stats.get("falldata_inline_stats", {})
        )

        temporal_stats = stats.get("falldata_temporal_stats", {})
        for field in (
            "requests",
            "disabled",
            "missing_dependencies",
            "process_calls",
            "process_completed",
            "process_errors",
            "process_timeouts",
            "process_seconds",
            # 재사용 프로세스에 보낸 요청이며 프로세스 시작 횟수와 구분한다.
            "worker_calls",
            "worker_completed",
            "worker_errors",
            "worker_timeouts",
            "worker_seconds",
            "parsed_results",
            "unparsed_results",
            "insufficient_pose_results",
            "window_inferences",
            "window_reports_missing",
            "insufficient_window_results",
        ):
            if field in temporal_stats:
                help_text = (
                    "TCN completed window calls explicitly reported by subprocess; "
                    "absent reports are counted separately, not inferred as one"
                    if field == "window_inferences"
                    else f"TCN subprocess {field}; not GPU inference or per-camera throughput"
                )
                family = CounterMetricFamily(
                    f"cctv_falldata_temporal_{field}",
                    help_text,
                )
                family.add_metric([], temporal_stats[field])
                yield family

        appearance_stats = stats.get("appearance_backend_stats", {})
        for backend_key in ("color_backend", "attribute_backend"):
            backend_fields = appearance_stats.get(backend_key, {})
            for field in (
                "requests",
                "completed",
                "errors",
                "seconds",
                "skipped_too_small",
            ):
                if field in backend_fields:
                    family = CounterMetricFamily(
                        f"cctv_appearance_{backend_key}_{field}",
                        f"Appearance {backend_key} Python {field}; call bookkeeping, "
                        "not color/attribute accuracy",
                    )
                    family.add_metric([], backend_fields[field])
                    yield family

        camera_stats = stats.get("primary_camera_stats", {})
        if not camera_stats:
            return
        for field_name in (
            "frames_seen",
            "frames_inferred",
            "frames_skipped",
            "frames_inference_unknown",
        ):
            family = CounterMetricFamily(
                f"cctv_camera_primary_{field_name}",
                f"Primary GIE observed {field_name}; not end-to-end AI throughput",
                labels=["camera_id"],
            )
            for camera_id, values in camera_stats.items():
                family.add_metric([camera_id], values[field_name])
            yield family
        age = GaugeMetricFamily(
            "cctv_camera_primary_last_frame_age_seconds",
            "Seconds since last primary GIE frame; absent until first observation",
            labels=["camera_id"],
        )
        for camera_id, values in camera_stats.items():
            seconds = values.get("last_frame_age_seconds")
            if seconds is not None:
                age.add_metric([camera_id], seconds)
        yield age

    @staticmethod
    def _collect_falldata_inline_metrics(snapshot):
        if not snapshot:
            return
        enabled = GaugeMetricFamily(
            "cctv_falldata_inline_enabled",
            "Inline RF enabled in configuration; not model readiness",
        )
        enabled.add_metric([], int(snapshot["enabled"]))
        yield enabled
        for field in (
            "requests",
            "ok_results",
            "unavailable_results",
            "error_results",
            "cache_hits",
            "predict_calls",
            "predict_completed",
            "predict_errors",
            "predict_seconds",
        ):
            family = CounterMetricFamily(
                f"cctv_falldata_inline_{field}",
                f"Verifier-lifetime inline RF {field}; not end-to-end AI throughput",
                labels=["camera_id"],
            )
            for camera, values in snapshot["cameras"].items():
                family.add_metric([camera], values.get(field, 0))
            yield family

    @staticmethod
    def _collect_model_metrics(snapshot):
        models = snapshot.get("models", {})
        if not models:
            return
        for field in ("pipeline_generation", "metadata_errors"):
            metric = GaugeMetricFamily(
                f"cctv_model_{field}", f"GIE observation {field}"
            )
            metric.add_metric([], snapshot[field])
            yield metric
        enabled = GaugeMetricFamily(
            "cctv_model_enabled",
            "GIE element present at pipeline build",
            labels=["model"],
        )
        for name, model in models.items():
            enabled.add_metric([name], int(model["enabled"]))
        yield enabled
        properties = GaugeMetricFamily(
            "cctv_model_build_property",
            "GObject value at pipeline build; absent if unreadable",
            labels=["model", "property"],
        )
        conflict = GaugeMetricFamily(
            "cctv_model_unique_id_conflict",
            "Duplicate GIE ID prevents model attribution",
            labels=["model"],
        )
        for name, model in models.items():
            conflict.add_metric([name], int(model["unique_id_conflict"]))
            for key, value in model["properties"].items():
                if value is not None:
                    properties.add_metric([name, key], value)
        yield properties
        yield conflict
        for field in (
            "frames_seen",
            "frames_tensor_output",
            "frames_no_objects",
            "frames_classifier_only",
            "frames_unobserved",
            "frames_unknown",
            "objects_with_tensor",
            "objects_classifier_only",
        ):
            metric = CounterMetricFamily(
                f"cctv_model_{field}",
                f"GIE metadata {field}; not GPU execution or end-to-end AI throughput",
                labels=["model", "camera_id"],
            )
            for name, model in models.items():
                for camera, counts in model["cameras"].items():
                    metric.add_metric([name, camera], counts[field])
            yield metric
        for kind in ("frame", "tensor"):
            field = f"last_{kind}_age_seconds"
            age = GaugeMetricFamily(
                f"cctv_model_{field}",
                f"Seconds since last observed {kind} at metadata boundary",
                labels=["model", "camera_id"],
            )
            for name, model in models.items():
                for camera, counts in model["cameras"].items():
                    if counts[field] is not None:
                        age.add_metric([name, camera], counts[field])
            yield age


def start_processor_metrics_server(processor: Any, port: int) -> Any:
    """백그라운드 Prometheus HTTP 서버를 시작하고 서버 핸들을 반환한다."""
    registry = CollectorRegistry(auto_describe=True)
    registry.register(ProcessorStatsCollector(processor))
    server, thread = start_http_server(port, registry=registry)
    logger.info("CCTV processor Prometheus metrics server started: port=%s", port)
    return server, thread

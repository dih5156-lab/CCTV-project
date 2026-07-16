from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


class AiotMetrics:
    def __init__(self, registry: CollectorRegistry | None = None):
        kwargs = {"registry": registry} if registry is not None else {}
        self.commands_received = Counter(
            "aiot_commands_received_total",
            "수신한 AIoT 명령 수",
            ["message_type"],
            **kwargs,
        )
        self.commands_duplicate = Counter(
            "aiot_commands_duplicate_total", "중복 AIoT 명령 수", **kwargs
        )
        self.commands_rejected = Counter(
            "aiot_commands_rejected_total",
            "거절한 AIoT 명령 수",
            ["reason"],
            **kwargs,
        )
        self.query_duration = Histogram(
            "aiot_query_duration_seconds",
            "AIoT 검색 처리 시간",
            ["search_mode", "status"],
            **kwargs,
        )
        self.query_matches = Counter(
            "aiot_query_matches_total",
            "AIoT 검색 결과 수",
            ["search_mode"],
            **kwargs,
        )
        self.query_inflight = Gauge(
            "aiot_query_inflight", "실행 중인 AIoT 검색 수", **kwargs
        )
        self.result_outbox_pending = Gauge(
            "aiot_result_outbox_pending", "대기 중인 AIoT 결과 수", **kwargs
        )
        self.result_retry = Counter(
            "aiot_result_retry_total", "AIoT 결과 재시도 수", **kwargs
        )
        self.media_upload_bytes = Counter(
            "aiot_media_upload_bytes_total", "업로드한 AIoT 미디어 바이트", **kwargs
        )
        self.media_upload_failures = Counter(
            "aiot_media_upload_failures_total",
            "AIoT 미디어 업로드 실패 수",
            ["reason"],
            **kwargs,
        )
        self.media_url_expired = Counter(
            "aiot_media_url_expired_total", "만료된 AIoT 업로드 URL 수", **kwargs
        )
        self.shadow_event_missing = Counter(
            "shadow_event_missing_total", "Shadow 경로 누락 이벤트 수", **kwargs
        )
        self.shadow_event_duplicate = Counter(
            "shadow_event_duplicate_total", "Shadow 경로 중복 이벤트 수", **kwargs
        )
        self.shadow_latency_delta = Histogram(
            "shadow_latency_delta_seconds", "기존 경로 대비 Shadow 지연 차이", **kwargs
        )

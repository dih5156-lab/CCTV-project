# EdgeX Bidirectional AIoT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 서버가 EdgeX를 통해 Jetson의 실시간·과거 AI 검색을 요청하고, 선택한 미디어만 임시 URL로 업로드받는 신뢰성 있는 양방향 AIoT 흐름을 구축한다.

**Architecture:** 기존 `CanonicalEvent`, `AppearanceLog`, `EdgeXDeviceAdapterService`, EdgeX Outbox를 유지하고 명령 계약, Command Inbox, 검색 서비스, 미디어 업로더를 작은 컴포넌트로 추가한다. 명령과 결과는 `request_id`로 연결하고 기존 MQTT 경로는 Shadow 기준선으로 유지한다.

**Tech Stack:** Python 3, SQLite, paho-mqtt, requests, EdgeX Foundry, Redis MessageBus, Prometheus, pytest, Docker Compose, Jetson DeepStream

## Global Constraints

- Jetson `docker-compose.jetson.yml` 운영 환경을 기준으로 한다.
- 새 외부 Python 라이브러리를 추가하지 않는다.
- `schema_version: "1.0"`과 기존 MQTT/Kuiper 필드를 유지한다.
- 이미지·영상 바이트를 EdgeX 이벤트에 포함하지 않는다.
- 서버가 요청한 `match_id`의 미디어만 요청별 HTTPS URL로 업로드한다.
- 기존 MQTT 경로는 Shadow 검증과 별도 운영 승인 전 제거하지 않는다.
- 실시간 추론과 중요 이벤트 처리가 검색·업로드보다 높은 우선순위를 가진다.
- 현재 사용자 수정이 있는 `src/core/deepstream_processor.py`와 `tests/test_deepstream_processor.py`는 이 계획의 초기 작업에서 수정하지 않는다.

---

## File Structure

- Create `src/aiot/contracts.py`: 명령 파싱·검증과 결과 envelope 생성
- Create `src/aiot/command_store.py`: SQLite Command Inbox와 멱등 상태 저장
- Create `src/aiot/query_service.py`: history/live/both 검색 조합과 match projection
- Create `src/aiot/media_uploader.py`: 미디어 경로·URL 검증 및 제한된 PUT 업로드
- Create `src/services/aiot_command_service.py`: 명령 오케스트레이션과 상태 게시
- Modify `src/edgex/adapter_service.py`: EdgeX command 구독과 서비스 lifecycle 연결
- Modify `runners/run_edgex_adapter.py`: AIoT 설정 전달
- Modify `src/canonical_event.py`: 선택적 `media`와 request correlation 필드 보강
- Modify `src/edgex/_outbox_mixin.py`: 결과 메시지의 멱등 키·목적지 저장
- Modify `src/api/v1/metrics.py`: AIoT 지표 추가
- Modify `.env.example`, `.env.jetson.example`, `docker-compose.jetson.yml`: 기능 플래그와 제한값
- Create `tests/test_aiot_contracts.py`, `tests/test_aiot_command_store.py`, `tests/test_aiot_query_service.py`, `tests/test_aiot_media_uploader.py`, `tests/test_aiot_command_service.py`
- Modify `tests/test_device_service.py`, `tests/test_canonical_event.py`, `tests/test_check_compose_runtime_assumptions.py`

### Task 1: Versioned AIoT Command Contracts

**Files:**
- Create: `src/aiot/__init__.py`
- Create: `src/aiot/contracts.py`
- Test: `tests/test_aiot_contracts.py`

**Interfaces:**
- Produces: `parse_ai_query_request(payload: Mapping[str, Any], now: datetime) -> AiQueryRequest`
- Produces: `parse_fetch_media_request(payload: Mapping[str, Any], now: datetime) -> FetchMediaRequest`
- Produces: `build_command_result(request_id: str, status: str, **fields: Any) -> dict[str, Any]`

- [ ] **Step 1: Write failing contract tests**

```python
def test_parse_ai_query_request_accepts_both_mode():
    request = parse_ai_query_request({
        "schema_version": "1.0", "message_type": "ai_query_request",
        "request_id": "q-1", "target": {"jetson_id": "edge-01"},
        "search_mode": "both", "filters": {"gender": "female", "has_handbag": True},
        "expires_at": "2099-01-01T00:00:00Z",
    }, now=datetime(2026, 7, 16, tzinfo=timezone.utc))
    assert request.request_id == "q-1"
    assert request.search_mode == "both"

def test_parse_request_rejects_expired_command():
    with pytest.raises(CommandValidationError, match="expired"):
        parse_ai_query_request(expired_payload(), now=datetime.now(timezone.utc))
```

- [ ] **Step 2: Verify RED**

Run: `rtk pytest tests/test_aiot_contracts.py -q`
Expected: FAIL because `src.aiot.contracts` does not exist.

- [ ] **Step 3: Implement immutable request models and strict validators**

```python
@dataclass(frozen=True)
class AiQueryRequest:
    request_id: str
    jetson_id: str
    camera_ids: tuple[str, ...]
    search_mode: Literal["live", "history", "both"]
    filters: Mapping[str, Any]
    query_text: str | None
    time_from: float | None
    time_to: float | None
    limit: int
    expires_at: datetime

def build_command_result(request_id: str, status: str, **fields: Any) -> dict[str, Any]:
    return {"schema_version": "1.0", "message_type": "ai_command_result",
            "request_id": request_id, "status": status, **fields}
```

Allow only documented filter keys and enforce `1 <= limit <= AIOT_QUERY_MAX_RESULTS`.

- [ ] **Step 4: Verify GREEN and commit**

Run: `rtk pytest tests/test_aiot_contracts.py -q`
Expected: PASS.

```bash
rtk git add src/aiot tests/test_aiot_contracts.py
rtk git commit -m "feat: add versioned AIoT command contracts"
```

### Task 2: Persistent Command Inbox

**Files:**
- Create: `src/aiot/command_store.py`
- Test: `tests/test_aiot_command_store.py`

**Interfaces:**
- Consumes: validated `request_id`, `message_type`, `expires_at`
- Produces: `CommandStore.claim(request_id, message_type, expires_at) -> ClaimResult`
- Produces: `CommandStore.update(request_id, status, result_payload=None) -> None`
- Produces: `CommandStore.get(request_id) -> CommandRecord | None`

- [ ] **Step 1: Write duplicate and restart tests**

```python
def test_claim_is_idempotent_across_reopen(tmp_path):
    path = tmp_path / "commands.db"
    assert CommandStore(path).claim("q-1", "ai_query_request", FUTURE).is_new
    assert not CommandStore(path).claim("q-1", "ai_query_request", FUTURE).is_new

def test_update_persists_completed_result(tmp_path):
    store = CommandStore(tmp_path / "commands.db")
    store.claim("q-1", "ai_query_request", FUTURE)
    store.update("q-1", "completed", {"matches": []})
    assert store.get("q-1").status == "completed"
```

- [ ] **Step 2: Verify RED**

Run: `rtk pytest tests/test_aiot_command_store.py -q`
Expected: FAIL because `CommandStore` is undefined.

- [ ] **Step 3: Implement SQLite schema and atomic claim**

```sql
CREATE TABLE IF NOT EXISTS aiot_command_inbox (
  request_id TEXT PRIMARY KEY, message_type TEXT NOT NULL,
  status TEXT NOT NULL, expires_at TEXT NOT NULL,
  result_json TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
```

Use `INSERT OR IGNORE`, a per-instance lock, WAL mode, and parameterized SQL. Never store `upload_url` in `result_json`.

- [ ] **Step 4: Verify GREEN and commit**

Run: `rtk pytest tests/test_aiot_command_store.py -q`
Expected: PASS.

```bash
rtk git add src/aiot/command_store.py tests/test_aiot_command_store.py
rtk git commit -m "feat: persist idempotent AIoT command state"
```

### Task 3: History and Live Query Service

**Files:**
- Create: `src/aiot/query_service.py`
- Test: `tests/test_aiot_query_service.py`

**Interfaces:**
- Consumes: `AiQueryRequest`, `AppearanceLog.search(...)`
- Produces: `AiQueryService.search(request: AiQueryRequest) -> list[dict[str, Any]]`
- Produces: injectable `LiveMatchProvider.search(filters, camera_ids, limit) -> list[Mapping[str, Any]]`

- [ ] **Step 1: Write history/both/dedup tests**

```python
def test_history_maps_handbag_and_gender_filters(appearance_log):
    service = AiQueryService(appearance_log, live_provider=FakeLive([]))
    matches = service.search(query(search_mode="history",
        filters={"gender": "female", "has_handbag": True}))
    assert matches[0]["attributes"]["gender"] == "female"

def test_both_deduplicates_same_crop_path(appearance_log):
    service = AiQueryService(appearance_log, FakeLive([ROW]))
    assert len(service.search(query(search_mode="both"))) == 1
```

- [ ] **Step 2: Verify RED**

Run: `rtk pytest tests/test_aiot_query_service.py -q`
Expected: FAIL because `AiQueryService` does not exist.

- [ ] **Step 3: Implement filter mapping and safe projection**

```python
history_rows = self.appearance_log.search(
    camera_id=single_camera, gender=filters.get("gender"),
    has_handbag=filters.get("has_handbag"),
    has_backpack=filters.get("has_backpack"),
    upper_color=filters.get("upper_color"), lower_color=filters.get("lower_color"),
    time_from=request.time_from, time_to=request.time_to, limit=request.limit,
)
```

Return only `match_id`, `camera_id`, `occurred_at`, `confidence`, `attributes`, `media_available`; do not return absolute paths. Initially inject `LiveMatchProvider`; do not modify the dirty DeepStream files.

- [ ] **Step 4: Verify GREEN and commit**

Run: `rtk pytest tests/test_aiot_query_service.py tests/test_appearance_log.py -q`
Expected: PASS.

```bash
rtk git add src/aiot/query_service.py tests/test_aiot_query_service.py
rtk git commit -m "feat: add local AIoT appearance query service"
```

### Task 4: Restricted On-Demand Media Upload

**Files:**
- Create: `src/aiot/media_uploader.py`
- Test: `tests/test_aiot_media_uploader.py`

**Interfaces:**
- Produces: `MediaUploader.upload(request: FetchMediaRequest, resolve_match: Callable[[str], Path | None]) -> list[UploadResult]`

- [ ] **Step 1: Write security and success tests**

```python
def test_rejects_non_https_url(uploader):
    with pytest.raises(MediaUploadError, match="https"):
        uploader.upload(fetch(upload_url="http://server/x"), resolver)

def test_uploads_only_requested_match(tmp_path, fake_session):
    result = uploader(fake_session).upload(fetch(match_ids=("m-1",)), resolver)
    assert fake_session.put.call_count == 1
    assert result[0].sha256
```

- [ ] **Step 2: Verify RED**

Run: `rtk pytest tests/test_aiot_media_uploader.py -q`
Expected: FAIL because `MediaUploader` does not exist.

- [ ] **Step 3: Implement validation and streaming PUT**

Validate HTTPS, exact allowlisted hostname, expiration, resolved path containment, file size, and `media_kind`. Stream with a timeout; return checksum and byte count. Never log query strings or authorization headers.

- [ ] **Step 4: Verify GREEN and commit**

Run: `rtk pytest tests/test_aiot_media_uploader.py -q`
Expected: PASS.

```bash
rtk git add src/aiot/media_uploader.py tests/test_aiot_media_uploader.py
rtk git commit -m "feat: upload requested AIoT media securely"
```

### Task 5: Command Orchestration and Reliable Results

**Files:**
- Create: `src/services/aiot_command_service.py`
- Modify: `src/edgex/_outbox_mixin.py`
- Test: `tests/test_aiot_command_service.py`
- Test: `tests/test_device_service.py`

**Interfaces:**
- Produces: `AiotCommandService.handle(payload: Mapping[str, Any]) -> None`
- Consumes: contract parsers, `CommandStore`, `AiQueryService`, `MediaUploader`, `publish_result(payload) -> bool`

- [ ] **Step 1: Write lifecycle, duplicate, and outage tests**

```python
def test_query_publishes_accepted_running_completed(service, publisher):
    service.handle(valid_query())
    assert [p["status"] for p in publisher.payloads] == ["accepted", "running", "completed"]

def test_duplicate_republishes_saved_result_without_search(service, query_service):
    service.handle(valid_query()); service.handle(valid_query())
    assert query_service.calls == 1

def test_failed_publish_is_written_to_outbox(service, outbox):
    service.publisher.return_value = False
    service.handle(valid_query())
    assert outbox.pending_count() >= 1
```

- [ ] **Step 2: Verify RED**

Run: `rtk pytest tests/test_aiot_command_service.py -q`
Expected: FAIL because the orchestrator does not exist.

- [ ] **Step 3: Implement orchestration and generic result outbox storage**

Keep physical-device `ActionBridge` unchanged. Add a generic `_store_failed_aiot_result(request_id, payload, last_error)` using `destination_type='edgex'`, `destination_name='aiot-command-result'`, and an idempotency-derived event ID.

- [ ] **Step 4: Verify GREEN and regression tests**

Run: `rtk pytest tests/test_aiot_command_service.py tests/test_device_service.py tests/test_action_bridge.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/services/aiot_command_service.py src/edgex/_outbox_mixin.py tests/test_aiot_command_service.py tests/test_device_service.py
rtk git commit -m "feat: orchestrate reliable EdgeX AIoT commands"
```

### Task 6: EdgeX Adapter Wiring and Feature Flags

**Files:**
- Modify: `src/edgex/adapter_service.py`
- Modify: `runners/run_edgex_adapter.py`
- Modify: `.env.example`
- Modify: `.env.jetson.example`
- Modify: `docker-compose.jetson.yml`
- Modify: `tests/test_check_compose_runtime_assumptions.py`
- Test: `tests/test_edgex_adapter_aiot_commands.py`

**Interfaces:**
- Command topic: `${AIOT_COMMAND_TOPIC_PREFIX:-edgex/commands/cctv}/<jetson_id>/#`
- Result resource/topic: `${AIOT_RESULT_TOPIC_PREFIX:-edgex/events/device}/<service>/aiot-command-result`

- [ ] **Step 1: Write disabled-by-default and routing tests**

```python
def test_aiot_commands_disabled_by_default(adapter):
    assert adapter.aiot_commands_enabled is False

def test_command_topic_routes_to_aiot_service(enabled_adapter):
    enabled_adapter._on_aiot_message(None, None, mqtt_message(valid_query()))
    enabled_adapter.aiot_command_service.handle.assert_called_once()
```

- [ ] **Step 2: Verify RED**

Run: `rtk pytest tests/test_edgex_adapter_aiot_commands.py -q`
Expected: FAIL because command routing is absent.

- [ ] **Step 3: Wire a separate command subscriber and lifecycle**

Add constructor settings for enable flag, Jetson ID, command topic, inbox DB, appearance DB, crop root, allowed upload hosts, max results, and max concurrent queries. Start/stop it with the adapter; malformed JSON must publish a rejected result when a request ID is recoverable.

- [ ] **Step 4: Add Compose configuration**

```yaml
AIOT_COMMANDS_ENABLED: ${AIOT_COMMANDS_ENABLED:-false}
AIOT_JETSON_ID: ${AIOT_JETSON_ID:-jetson-01}
AIOT_COMMAND_TOPIC_PREFIX: ${AIOT_COMMAND_TOPIC_PREFIX:-edgex/commands/cctv}
AIOT_ALLOWED_UPLOAD_HOSTS: ${AIOT_ALLOWED_UPLOAD_HOSTS:-}
AIOT_QUERY_MAX_RESULTS: ${AIOT_QUERY_MAX_RESULTS:-20}
AIOT_QUERY_MAX_CONCURRENT: ${AIOT_QUERY_MAX_CONCURRENT:-1}
AIOT_COMMAND_DB: /app/data/runtime/aiot_commands.db
```

- [ ] **Step 5: Verify config and tests**

Run: `rtk pytest tests/test_edgex_adapter_aiot_commands.py tests/test_check_compose_runtime_assumptions.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
rtk git add src/edgex/adapter_service.py runners/run_edgex_adapter.py .env.example .env.jetson.example docker-compose.jetson.yml tests/test_edgex_adapter_aiot_commands.py tests/test_check_compose_runtime_assumptions.py
rtk git commit -m "feat: wire AIoT commands into EdgeX adapter"
```

### Task 7: Canonical Projection, Metrics, and Shadow Verification

**Files:**
- Modify: `src/canonical_event.py`
- Modify: `src/api/v1/metrics.py`
- Modify: `tests/test_canonical_event.py`
- Create: `tests/test_aiot_shadow_flow.py`
- Modify: `scripts/health/check_jetson_edgex_stack.py`

**Interfaces:**
- Produces: `project_edgex_event(canonical: Mapping[str, Any]) -> dict[str, Any]`
- Produces metrics named in the design spec, without high-cardinality request IDs as labels

- [ ] **Step 1: Write projection and end-to-end Shadow tests**

```python
def test_projection_excludes_raw_and_keeps_media_reference():
    projected = project_edgex_event(canonical_with_raw_and_media())
    assert "raw" not in projected
    assert projected["snapshot_url"].endswith("event-1.jpg")

def test_shadow_query_survives_broker_outage(shadow_stack):
    shadow_stack.disconnect_results()
    shadow_stack.send_query("q-1")
    assert shadow_stack.outbox_pending() > 0
    shadow_stack.reconnect_results()
    assert shadow_stack.wait_completed("q-1")
```

- [ ] **Step 2: Verify RED**

Run: `rtk pytest tests/test_canonical_event.py tests/test_aiot_shadow_flow.py -q`
Expected: FAIL because projection and Shadow harness are absent.

- [ ] **Step 3: Implement projection and bounded metrics**

Add counters/histograms/gauges for commands, query duration/matches/inflight, result retries/pending, upload bytes/failures/expiry, and Shadow missing/duplicate/latency. Labels may include status and search mode, never `request_id`, `camera_id`, URL, or filename.

- [ ] **Step 4: Extend health check**

Check command subscriber connectivity, Inbox DB writability, Outbox pending count, allowed host configuration, and feature-flag state. Report disabled as `SKIP`, not failure.

- [ ] **Step 5: Run full verification**

Run: `rtk ruff check src/aiot src/services/aiot_command_service.py src/edgex src/api/v1/metrics.py`
Expected: no lint errors.

Run: `rtk pytest tests/test_aiot_contracts.py tests/test_aiot_command_store.py tests/test_aiot_query_service.py tests/test_aiot_media_uploader.py tests/test_aiot_command_service.py tests/test_edgex_adapter_aiot_commands.py tests/test_aiot_shadow_flow.py tests/test_canonical_event.py tests/test_device_service.py tests/test_action_bridge.py -q`
Expected: PASS.

Run: `rtk docker compose -f docker-compose.jetson.yml config -q`
Expected: exit 0.

- [ ] **Step 6: Commit**

```bash
rtk git add src/canonical_event.py src/api/v1/metrics.py tests/test_canonical_event.py tests/test_aiot_shadow_flow.py scripts/health/check_jetson_edgex_stack.py
rtk git commit -m "feat: observe and verify Shadow AIoT delivery"
```

### Task 8: Jetson Pilot Gate

**Files:**
- Create: `docs/operations/edgex-aiot-pilot.md`
- Modify: `scripts/ops/run_operation_check.sh`

**Interfaces:**
- Consumes: all prior tasks
- Produces: repeatable Mirror → Query Pilot → Media Pilot procedure and rollback command

- [ ] **Step 1: Document exact pilot sequence**

Include: enable Mirror with `AIOT_COMMANDS_ENABLED=false`; capture baseline FPS; enable one allowlisted Jetson/camera; issue one `history`, one `live`, one `both` request; request one snapshot with a short-lived URL; disconnect/reconnect EdgeX; confirm Outbox drains; disable the feature flag for rollback.

- [ ] **Step 2: Add operation-check assertions**

The operation check must fail on important-event loss, unbounded Outbox growth, unauthorized upload host, cross-request result mismatch, or FPS below the configured floor.

- [ ] **Step 3: Run Jetson validation**

Run: `rtk .venv/bin/python scripts/health/check_jetson_edgex_stack.py`
Expected: existing checks PASS and AIoT checks show PASS when enabled or SKIP when disabled.

Run the repository's approved DeepStream stability watch with one camera and compare baseline vs query/upload FPS. Expected: no important-event loss, no process restart, Outbox returns to zero after reconnection, and FPS stays above the configured floor.

- [ ] **Step 4: Commit pilot documentation and checks**

```bash
rtk git add docs/operations/edgex-aiot-pilot.md scripts/ops/run_operation_check.sh scripts/health/check_jetson_edgex_stack.py
rtk git commit -m "docs: add EdgeX AIoT Jetson pilot gate"
```

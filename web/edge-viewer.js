"use strict";

const EDGE_STATES = {
  unassigned: "담당 없음", preparing: "첫 담당 준비 중", reconciling: "제어기 재확인 필요",
  unavailable: "현재 실행 확인 불가", awaiting_results: "최근 결과 저장 대기",
  running: "추론·결과 저장 확인", switching: "기존 담당 유지 · 다음 담당 준비 중",
};
const EDGE_REASONS = {
  controller_reconciling: "제어기 재시작 후 실행을 다시 확인해야 합니다.",
  node_not_observed: "담당 노드의 최근 보고를 확인할 수 없습니다.",
  node_observation_stale: "담당 노드의 보고가 오래됐습니다.",
  observation_predates_controller: "제어기 시작 이후의 보고를 기다립니다.",
  node_not_registered: "담당 노드가 등록되어 있지 않습니다.",
  owner_execution_unverified: "배정된 AI 실행과 현재 보고가 일치하지 않습니다.",
  recent_result_missing: "현재 실행의 최근 결과 저장을 기다립니다.",
};

function edgeViewSummary(view, cameraId) {
  if (!view || view.camera_id !== cameraId || !Object.hasOwn(EDGE_STATES, view.status)
      || typeof view.inference_ready !== "boolean" || view.media_verified !== false
      || !Array.isArray(view.history) || view.history.length > 20) {
    throw new Error("invalid_view");
  }
  if (view.inference_ready !== ["running", "switching"].includes(view.status)
      || (view.inference_ready && (!view.active || !view.latest_result || !view.accepting_results))) {
    throw new Error("inconsistent_view");
  }
  return {
    state: EDGE_STATES[view.status],
    owner: view.active ? view.active.node_id : "—",
    epoch: view.active ? String(view.active.epoch) : "—",
    candidate: view.pending ? view.pending.node_id : "—",
    reason: EDGE_REASONS[view.reason] || (view.reason ? "상태를 다시 확인해 주세요." : "정상 조회 중입니다."),
    result: view.inference_ready && view.latest_result
      ? (view.latest_result.kind === "observation" ? "추론 상태 표본 저장 확인" : "감지 이벤트 저장 확인")
      : "현재 실행의 최근 저장 미확인",
  };
}

function mountEdgeViewer(doc, request = fetch, scheduler = globalThis) {
  const element = (id) => doc.getElementById(id);
  let generation = 0;
  let timer = null;
  let controller = null;
  let token = "";
  let previewTimer = null;
  let lastPreview = null;

  function clearPreview(message = "미연결") {
    scheduler.clearTimeout(previewTimer);
    element("preview-image").hidden = true;
    element("preview-image").removeAttribute("src");
    element("preview-status").textContent = message;
  }

  function emptyHistory(message) {
    const row = doc.createElement("tr");
    const cell = doc.createElement("td");
    cell.colSpan = 5;
    cell.textContent = message;
    row.appendChild(cell);
    return row;
  }

  function clearView(message) {
    clearPreview();
    lastPreview = null;
    for (const id of ["state", "owner", "epoch", "candidate", "updated", "result"]) {
      element(id).textContent = id === "state" || id === "result" ? "미확인" : "—";
    }
    element("history").replaceChildren(emptyHistory("확인된 이력이 없습니다."));
    element("connection").textContent = message;
  }

  function disconnect(message = "연결을 해제했습니다.") {
    generation += 1;
    scheduler.clearTimeout(timer);
    if (controller) controller.abort();
    controller = null;
    token = "";
    element("reader-token").value = "";
    clearView(message);
  }

  function show(view, cameraId) {
    const summary = edgeViewSummary(view, cameraId);
    for (const key of ["state", "owner", "epoch", "candidate", "result"]) element(key).textContent = summary[key];
    element("connection").textContent = summary.reason;
    element("updated").textContent = view.observed_at;
    const labels = { preparing: "준비 중", active: "현재 담당", retired: "이전 담당", aborted: "취소" };
    const rows = view.history.map((item) => {
      const row = doc.createElement("tr");
      for (const value of [item.epoch, item.node_id, labels[item.state] || "미확인", item.prepared_at || "—", item.activated_at || "—"]) {
        const cell = doc.createElement("td");
        cell.textContent = String(value);
        row.appendChild(cell);
      }
      return row;
    });
    element("history").replaceChildren(...(rows.length ? rows : [emptyHistory("배정 이력이 없습니다.")]));
    if (!view.inference_ready || !view.preview_available || (lastPreview && lastPreview.assignment !== view.active.assignment_id)) {
      clearPreview(view.preview_available ? "현재 담당의 새 프레임 대기" : "미리보기 끝점이 연결되지 않았습니다.");
      lastPreview = null;
    }
  }

  async function preview(view, cameraId, expectedGeneration, signal) {
    if (!view.inference_ready || !view.preview_available) return;
    const started = performance.now();
    try {
      const response = await request(`/v1/cameras/${encodeURIComponent(cameraId)}/preview`, {
        headers: { Authorization: `Bearer ${token}`, "X-Edge-Assignment-Id": view.active.assignment_id },
        signal, cache: "no-store", redirect: "error", credentials: "omit",
      });
      if (!response.ok) throw new Error("preview_unavailable");
      const payload = await response.json();
      if (generation !== expectedGeneration) return;
      const checked = edgePreviewImage(payload, view.active, lastPreview);
      const decoded = new Image();
      decoded.src = checked.url;
      await decoded.decode();
      if (generation !== expectedGeneration) return;
      const remaining = 2000 - payload.age_seconds * 1000 - (performance.now() - started);
      if (remaining <= 0) throw new Error("preview_expired");
      clearPreview();
      element("preview-image").src = checked.url;
      element("preview-image").hidden = false;
      element("preview-status").textContent = "담당 영상의 새 프레임 확인 · 2초 간격 미리보기";
      lastPreview = checked;
      previewTimer = scheduler.setTimeout(() => clearPreview("새 프레임 갱신 대기"), remaining);
    } catch (_) {
      if (generation === expectedGeneration) clearPreview("담당 영상 확인 불가 · 다시 연결 중");
    }
  }

  async function poll(cameraId, expectedGeneration) {
    if (generation !== expectedGeneration) return;
    const activeController = new AbortController();
    controller = activeController;
    const timeout = scheduler.setTimeout(() => activeController.abort(), 3000);
    let retry = true;
    try {
      const response = await request(`/v1/cameras/${encodeURIComponent(cameraId)}/view`, {
        headers: { Authorization: `Bearer ${token}` }, signal: activeController.signal,
        cache: "no-store", redirect: "error", credentials: "omit",
      });
      if (generation !== expectedGeneration) return;
      if (response.status === 401) {
        retry = false;
        disconnect("인증에 실패했습니다. 읽기 전용 토큰을 다시 입력해 주세요.");
        return;
      }
      if (!response.ok) throw new Error("request_failed");
      const view = await response.json();
      // 카메라 변경/연결 해제 전에 보낸 응답이 새 화면을 덮어쓰지 않게 한다.
      if (generation === expectedGeneration) {
        show(view, cameraId);
        await preview(view, cameraId, expectedGeneration, activeController.signal);
      }
    } catch (_) {
      if (generation === expectedGeneration) clearView("상태 조회에 실패했습니다. 다시 연결 중입니다.");
    } finally {
      scheduler.clearTimeout(timeout);
      if (retry && generation === expectedGeneration) timer = scheduler.setTimeout(() => poll(cameraId, expectedGeneration), 2000);
    }
  }

  element("connect-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const cameraId = element("camera-id").value.trim();
    const nextToken = element("reader-token").value.trim();
    disconnect("상태를 조회하고 있습니다.");
    if (!/^[A-Za-z0-9_.-]{1,128}$/.test(cameraId) || !/^[A-Za-z0-9_-]{32,128}$/.test(nextToken)) {
      clearView("카메라 ID와 읽기 전용 토큰을 확인해 주세요.");
      return;
    }
    token = nextToken;
    poll(cameraId, generation);
  });
  element("disconnect").addEventListener("click", () => disconnect());
  return { disconnect };
}

function edgePreviewImage(payload, owner, previous) {
  const fields = ["node_id", "camera_id", "assignment_id", "epoch", "supervisor_boot_id", "session_id", "activation_id"];
  if (!owner || fields.some((key) => key === "epoch" ? !Number.isSafeInteger(owner[key]) || owner[key] <= 0 : typeof owner[key] !== "string" || !owner[key])
      || !payload || !payload.edge || fields.some((key) => payload.edge[key] !== owner[key])
      || !payload.frame || payload.frame.camera_id !== owner.camera_id
      || !Number.isSafeInteger(payload.frame.frame_token) || payload.frame.frame_token <= 0
      || !Number.isSafeInteger(payload.frame.generation) || payload.frame.generation < 0
      || !Number.isFinite(payload.age_seconds) || payload.age_seconds < 0 || payload.age_seconds >= 2
      || typeof payload.jpeg_base64 !== "string" || payload.jpeg_base64.length > 699052
      || !/^[A-Za-z0-9+/]+={0,2}$/.test(payload.jpeg_base64)) throw new Error("invalid_preview");
  const current = { assignment: owner.assignment_id, generation: payload.frame.generation, sequence: payload.frame.frame_token,
    url: "data:image/jpeg;base64," + payload.jpeg_base64 };
  if (previous && previous.assignment === current.assignment && (current.generation < previous.generation
      || (current.generation === previous.generation && current.sequence <= previous.sequence))) throw new Error("preview_not_advancing");
  return current;
}

if (typeof document !== "undefined") {
  const viewer = mountEdgeViewer(document);
  addEventListener("pagehide", () => viewer.disconnect());
}
if (typeof module !== "undefined") module.exports = { edgeViewSummary, edgePreviewImage, mountEdgeViewer };

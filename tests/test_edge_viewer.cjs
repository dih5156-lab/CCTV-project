const test = require("node:test");
const assert = require("node:assert/strict");
const { edgeViewSummary, edgePreviewImage, mountEdgeViewer } = require("../web/edge-viewer.js");

function view(camera = "camera_1", node = "jetson-a") {
  return { camera_id: camera, status: "running", inference_ready: true, media_verified: false,
    accepting_results: true, active: { node_id: node, epoch: 1 }, pending: null,
    latest_result: { kind: "observation" }, observed_at: "2026-10-23T06:00:00Z", reason: null,
    history: [{ node_id: node, epoch: 1, state: "active", prepared_at: "now", activated_at: "now" }] };
}

function page() {
  class Element {
    constructor() { this.value = ""; this.textContent = ""; this.children = []; this.listeners = {}; }
    appendChild(child) { this.children.push(child); }
    replaceChildren(...children) { this.children = children; }
    addEventListener(name, listener) { this.listeners[name] = listener; }
    removeAttribute(name) { delete this[name]; }
    set innerHTML(_) { throw new Error("HTML injection is forbidden"); }
  }
  const elements = {};
  const doc = { getElementById: (id) => elements[id] ||= new Element(), createElement: () => new Element() };
  const timers = new Map();
  let counter = 0;
  const scheduler = { setTimeout(callback, delay) { timers.set(++counter, { callback, delay }); return counter; }, clearTimeout(id) { timers.delete(id); } };
  const requests = [];
  const fetch = (url, options) => new Promise((resolve, reject) => requests.push({ url, options, resolve, reject }));
  const mounted = mountEdgeViewer(doc, fetch, scheduler);
  const connect = (camera = "camera_1") => {
    doc.getElementById("camera-id").value = camera;
    doc.getElementById("reader-token").value = "r".repeat(40);
    elements["connect-form"].listeners.submit({ preventDefault() {} });
  };
  const reply = (request, body, status = 200) => request.resolve({ status, ok: status === 200, json: async () => body });
  const flush = () => new Promise((resolve) => setImmediate(resolve));
  return { elements, timers, scheduler, requests, mounted, connect, reply, flush };
}

test("추론 표본 확인을 영상 재생 확인으로 바꾸지 않는다", () => {
  assert.equal(edgeViewSummary(view(), "camera_1").result, "추론 상태 표본 저장 확인");
  assert.throws(() => edgeViewSummary({ ...view(), media_verified: true }, "camera_1"));
  assert.throws(() => edgeViewSummary({ ...view(), inference_ready: false }, "camera_1"));
  assert.throws(() => edgeViewSummary(view(), "camera_2"));
});

test("토큰은 URL에 넣지 않고 입력란을 비우며 이력은 텍스트로 표시한다", async () => {
  const p = page(); p.connect();
  assert.equal(p.elements["reader-token"].value, "");
  assert.equal(p.requests[0].url, "/v1/cameras/camera_1/view");
  assert.equal(p.requests[0].options.headers.Authorization, "Bearer " + "r".repeat(40));
  assert.equal(p.requests[0].options.redirect, "error");
  const unsafe = '<img src=x onerror="alert(1)">';
  p.reply(p.requests[0], view("camera_1", unsafe)); await p.flush();
  assert.equal(p.elements.owner.textContent, unsafe);
  assert.equal(p.elements.history.children[0].children[1].textContent, unsafe);
});

test("이전 카메라 응답과 타임아웃이 새 연결에 영향을 주지 않는다", async () => {
  const p = page(); p.connect();
  const firstTimeout = [...p.timers.values()].find((t) => t.delay === 3000).callback;
  p.connect("camera_2");
  assert.equal(p.requests[0].options.signal.aborted, true);
  firstTimeout();
  assert.equal(p.requests[1].options.signal.aborted, false);
  p.reply(p.requests[1], view("camera_2", "jetson-b")); await p.flush();
  p.reply(p.requests[0], view()); await p.flush();
  assert.equal(p.elements.owner.textContent, "jetson-b");
  assert.equal([...p.timers.values()].filter((t) => t.delay === 2000).length, 1);
});

test("통신 실패 뒤에는 이전 정상 상태와 이력을 지운다", async () => {
  const p = page(); p.connect(); p.reply(p.requests[0], view()); await p.flush();
  [...p.timers.values()].find((t) => t.delay === 2000).callback();
  p.requests[1].reject(new Error("offline")); await p.flush();
  assert.equal(p.elements.state.textContent, "미확인");
  assert.equal(p.elements.owner.textContent, "—");
  assert.equal(p.elements.history.children[0].children[0].colSpan, 5);
});

test("인증 실패와 연결 해제 뒤에는 재시도하지 않는다", async () => {
  const p = page(); p.connect(); p.reply(p.requests[0], {}, 401); await p.flush();
  assert.equal(p.timers.size, 0);
  p.connect(); p.mounted.disconnect();
  p.reply(p.requests[1], view()); await p.flush();
  assert.equal(p.elements.owner.textContent, "—");
  assert.equal(p.timers.size, 0);
});

test("빈 이력 응답은 앞선 담당 이력을 남기지 않는다", async () => {
  const p = page(); p.connect(); p.reply(p.requests[0], view()); await p.flush();
  [...p.timers.values()].find((t) => t.delay === 2000).callback();
  p.reply(p.requests[1], { ...view(), status: "unassigned", inference_ready: false, active: null, latest_result: null, history: [] });
  await p.flush();
  assert.equal(p.elements.owner.textContent, "—");
  assert.equal(p.elements.history.children[0].children[0].textContent, "배정 이력이 없습니다.");
});

test("영상은 현재 실행·새 프레임·유효 나이가 함께 맞아야 한다", () => {
  const owner = { node_id: "jetson-a", camera_id: "camera_1", assignment_id: "assignment", epoch: 1,
    supervisor_boot_id: "boot", session_id: "session", activation_id: "activation" };
  const payload = { edge: owner, frame: {camera_id: "camera_1", generation: 0, frame_token: 2}, age_seconds: .1, jpeg_base64: "/9j/2Q==" };
  const first = edgePreviewImage(payload, owner, null);
  assert.match(first.url, /^data:image\/jpeg;base64,/);
  assert.throws(() => edgePreviewImage(payload, owner, first));
  assert.throws(() => edgePreviewImage({...payload, age_seconds: 2}, owner, null));
  assert.throws(() => edgePreviewImage({...payload, edge: {...owner, session_id: "old"}}, owner, null));
  assert.throws(() => edgePreviewImage({...payload, jpeg_base64: "javascript:alert(1)"}, owner, null));
  assert.throws(() => edgePreviewImage({...payload, edge: {}}, {}, null));
  assert.equal(edgePreviewImage({...payload, frame: {...payload.frame, frame_token: 3}}, owner, first).sequence, 3);
});

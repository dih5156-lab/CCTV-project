"""Local API documentation page helpers."""

from __future__ import annotations


def local_api_docs_html() -> str:
    """Return a small OpenAPI explorer that does not depend on external CDNs."""
    return """
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CCTV Platform API Docs</title>
  <style>
    :root {
      color-scheme: light dark;
      --bg: #f7f8fb;
      --panel: #ffffff;
      --text: #151923;
      --muted: #5b6472;
      --line: #d9dee8;
      --accent: #2563eb;
      --get: #0f766e;
      --post: #1d4ed8;
      --delete: #b91c1c;
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --bg: #10141b;
        --panel: #171d27;
        --text: #eef2f7;
        --muted: #aab3c2;
        --line: #2a3341;
        --accent: #7aa2ff;
      }
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }
    main {
      width: min(1120px, calc(100% - 32px));
      margin: 0 auto;
      padding: 32px 0 48px;
    }
    header {
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 20px;
      border-bottom: 1px solid var(--line);
      padding-bottom: 20px;
      margin-bottom: 24px;
    }
    h1 { margin: 0; font-size: 28px; }
    p { margin: 8px 0 0; color: var(--muted); }
    a { color: var(--accent); text-decoration: none; }
    code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .toolbar {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
      margin-bottom: 16px;
    }
    input {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      color: var(--text);
      padding: 10px 12px;
      font-size: 15px;
    }
    button, .link-button {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      color: var(--text);
      padding: 10px 12px;
      font-size: 14px;
      cursor: pointer;
      white-space: nowrap;
    }
    .endpoint {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      margin-bottom: 10px;
      overflow: hidden;
    }
    .endpoint summary {
      display: grid;
      grid-template-columns: 86px 1fr;
      gap: 12px;
      align-items: center;
      padding: 12px 14px;
      cursor: pointer;
    }
    .method {
      display: inline-flex;
      justify-content: center;
      align-items: center;
      min-height: 28px;
      border-radius: 5px;
      color: white;
      font-weight: 700;
      font-size: 13px;
    }
    .GET { background: var(--get); }
    .POST { background: var(--post); }
    .DELETE { background: var(--delete); }
    .path { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; overflow-wrap: anywhere; }
    .details {
      border-top: 1px solid var(--line);
      padding: 12px 14px 14px;
    }
    pre {
      overflow: auto;
      background: var(--bg);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 12px;
      font-size: 13px;
    }
    .empty, .error {
      padding: 18px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      color: var(--muted);
    }
    @media (max-width: 640px) {
      header, .toolbar { grid-template-columns: 1fr; display: grid; align-items: start; }
      .endpoint summary { grid-template-columns: 72px 1fr; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>CCTV Platform API</h1>
        <p>로컬 OpenAPI 문서입니다. 외부 CDN 없이 <code>/openapi.json</code>만 사용합니다.</p>
      </div>
      <a class="link-button" href="/openapi.json">OpenAPI JSON</a>
    </header>
    <section class="toolbar">
      <input id="filter" type="search" placeholder="경로, 태그, 설명 검색" autocomplete="off">
      <button id="reload" type="button">Reload</button>
    </section>
    <section id="status" class="empty">API 문서를 불러오는 중입니다...</section>
    <section id="endpoints"></section>
  </main>
  <script>
    const endpointsEl = document.getElementById("endpoints");
    const filterEl = document.getElementById("filter");
    const statusEl = document.getElementById("status");
    const reloadEl = document.getElementById("reload");
    let endpoints = [];

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    function endpointMatches(endpoint, query) {
      if (!query) return true;
      const text = [
        endpoint.method,
        endpoint.path,
        endpoint.summary,
        endpoint.description,
        endpoint.tags.join(" ")
      ].join(" ").toLowerCase();
      return text.includes(query.toLowerCase());
    }

    function render() {
      const query = filterEl.value.trim();
      const visible = endpoints.filter((endpoint) => endpointMatches(endpoint, query));
      endpointsEl.innerHTML = "";
      statusEl.hidden = visible.length > 0;
      statusEl.textContent = endpoints.length === 0
        ? "표시할 API 엔드포인트가 없습니다."
        : "검색 결과가 없습니다.";

      for (const endpoint of visible) {
        const details = document.createElement("details");
        details.className = "endpoint";
        const operation = {
          summary: endpoint.summary,
          description: endpoint.description,
          tags: endpoint.tags,
          parameters: endpoint.parameters,
          requestBody: endpoint.requestBody,
          responses: endpoint.responses,
        };
        details.innerHTML = `
          <summary>
            <span class="method ${escapeHtml(endpoint.method)}">${escapeHtml(endpoint.method)}</span>
            <span>
              <strong class="path">${escapeHtml(endpoint.path)}</strong>
              <span>${endpoint.summary ? " - " + escapeHtml(endpoint.summary) : ""}</span>
            </span>
          </summary>
          <div class="details">
            <pre>${escapeHtml(JSON.stringify(operation, null, 2))}</pre>
          </div>
        `;
        endpointsEl.appendChild(details);
      }
    }

    async function loadOpenApi() {
      statusEl.hidden = false;
      statusEl.className = "empty";
      statusEl.textContent = "API 문서를 불러오는 중입니다...";
      endpointsEl.innerHTML = "";
      try {
        const response = await fetch("/openapi.json", { cache: "no-store" });
        if (!response.ok) {
          throw new Error(`/openapi.json 응답 실패: ${response.status}`);
        }
        const spec = await response.json();
        endpoints = Object.entries(spec.paths || {}).flatMap(([path, methods]) =>
          Object.entries(methods).map(([method, operation]) => ({
            path,
            method: method.toUpperCase(),
            summary: operation.summary || "",
            description: operation.description || "",
            tags: operation.tags || [],
            parameters: operation.parameters || [],
            requestBody: operation.requestBody || null,
            responses: operation.responses || {},
          }))
        );
        render();
      } catch (error) {
        statusEl.className = "error";
        statusEl.hidden = false;
        statusEl.textContent = `문서 로드 실패: ${error.message}`;
      }
    }

    filterEl.addEventListener("input", render);
    reloadEl.addEventListener("click", loadOpenApi);
    loadOpenApi();
  </script>
</body>
</html>
    """.strip()

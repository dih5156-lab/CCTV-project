# Per-Camera RTSP Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish each active camera's OSD-annotated DeepStream output to an RTSP/WebRTC path derived from its camera ID.

**Architecture:** Keep the existing `nvstreammux` batch inference, tracking, and OSD path. In `rtsp-publish` mode, add `nvstreamdemux` after OSD (or after the preview tee output queue), then connect each demux source pad to a camera-specific queue, H.264 hardware encoding chain, and `rtspclientsink`. Resolve and validate output URLs in a focused helper so legacy single-camera configuration remains explicit and testable.

**Tech Stack:** Python 3.10, NVIDIA DeepStream/GStreamer, MediaMTX, Docker Compose, pytest, Ruff

## Global Constraints

- Preserve the existing batched inference, tracker, event probe, and preview behavior.
- RTSP output contains the `nvdsosd` analysis overlay during the POC phase.
- Default template is exactly `rtsp://cctv-media-server:8554/{camera_id}`.
- Camera IDs may contain only ASCII letters, digits, `_`, and `-`.
- A single active camera may continue using legacy `DS_RTSP_LOCATION` when no template is configured.
- Two or more active cameras must reject a legacy single URL when no template is configured.
- Do not add a third-party dependency.
- Keep non-RTSP output modes on their current single-output path.
- Use a separate hardware H.264 encoder branch per active camera.
- Runtime camera add/remove continues to use the existing pipeline restart behavior.

---

### Task 1: Resolve and validate camera-specific RTSP locations

**Files:**
- Create: `src/core/_deepstream_rtsp_output.py`
- Create: `tests/test_deepstream_rtsp_output.py`

**Interfaces:**
- Consumes: ordered `camera_ids: Sequence[str]`, optional `location_template`, optional `legacy_location`.
- Produces: `resolve_rtsp_locations(camera_ids, *, location_template, legacy_location) -> dict[str, str]` and `validate_camera_id(camera_id: str) -> None`.

- [ ] **Step 1: Write failing URL-resolution tests**

```python
import pytest

from src.core._deepstream_rtsp_output import resolve_rtsp_locations


def test_resolve_rtsp_locations_expands_camera_id_template():
    assert resolve_rtsp_locations(
        ["camera_1", "entrance-2"],
        location_template="rtsp://media:8554/{camera_id}",
        legacy_location=None,
    ) == {
        "camera_1": "rtsp://media:8554/camera_1",
        "entrance-2": "rtsp://media:8554/entrance-2",
    }


def test_resolve_rtsp_locations_keeps_single_camera_legacy_url():
    assert resolve_rtsp_locations(
        ["camera_1"],
        location_template=None,
        legacy_location="rtsp://media:8554/existing",
    ) == {"camera_1": "rtsp://media:8554/existing"}


def test_resolve_rtsp_locations_rejects_legacy_url_for_multiple_cameras():
    with pytest.raises(ValueError, match="DS_RTSP_LOCATION_TEMPLATE"):
        resolve_rtsp_locations(
            ["camera_1", "camera_2"],
            location_template=None,
            legacy_location="rtsp://media:8554/shared",
        )


@pytest.mark.parametrize("camera_id", ["camera/1", "camera 1", "카메라1", ""])
def test_resolve_rtsp_locations_rejects_unsafe_camera_id(camera_id):
    with pytest.raises(ValueError, match="camera ID"):
        resolve_rtsp_locations(
            [camera_id],
            location_template="rtsp://media:8554/{camera_id}",
            legacy_location=None,
        )
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `rtk pytest tests/test_deepstream_rtsp_output.py -q`

Expected: FAIL during collection because `src.core._deepstream_rtsp_output` does not exist.

- [ ] **Step 3: Implement the minimal resolver**

```python
from __future__ import annotations

import re
from collections.abc import Sequence

DEFAULT_RTSP_LOCATION_TEMPLATE = "rtsp://cctv-media-server:8554/{camera_id}"
_CAMERA_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def validate_camera_id(camera_id: str) -> None:
    if not isinstance(camera_id, str) or not _CAMERA_ID_PATTERN.fullmatch(camera_id):
        raise ValueError(
            f"유효하지 않은 camera ID: {camera_id!r}; 영문자, 숫자, _, -만 사용할 수 있습니다."
        )


def resolve_rtsp_locations(
    camera_ids: Sequence[str],
    *,
    location_template: str | None,
    legacy_location: str | None,
) -> dict[str, str]:
    ids = list(camera_ids)
    for camera_id in ids:
        validate_camera_id(camera_id)

    template = location_template.strip() if location_template else None
    legacy = legacy_location.strip() if legacy_location else None
    if template:
        if "{camera_id}" not in template:
            raise ValueError("DS_RTSP_LOCATION_TEMPLATE에 {camera_id}가 필요합니다.")
        return {camera_id: template.replace("{camera_id}", camera_id) for camera_id in ids}
    if legacy:
        if len(ids) != 1:
            raise ValueError(
                "다중 카메라 RTSP 출력에는 DS_RTSP_LOCATION_TEMPLATE이 필요합니다."
            )
        return {ids[0]: legacy}
    return {
        camera_id: DEFAULT_RTSP_LOCATION_TEMPLATE.replace("{camera_id}", camera_id)
        for camera_id in ids
    }
```

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run: `rtk pytest tests/test_deepstream_rtsp_output.py -q`

Expected: all tests PASS.

- [ ] **Step 5: Commit the resolver**

```bash
rtk git add src/core/_deepstream_rtsp_output.py tests/test_deepstream_rtsp_output.py
rtk git commit -m "Add per-camera RTSP location resolver"
```

### Task 2: Build and link one demux output branch per camera

**Files:**
- Modify: `src/core/_deepstream_rtsp_output.py`
- Modify: `src/core/_deepstream_pipeline_builder.py`
- Modify: `tests/test_deepstream_rtsp_output.py`
- Modify: `tests/test_deepstream_processor.py`

**Interfaces:**
- Consumes: source entries shaped as `list[tuple[int, str, dict[str, Any], str]]`, resolved location map, existing GStreamer element factory callbacks.
- Produces: `RtspOutputBranch`, `create_rtsp_output_branches(...) -> tuple[Any, list[RtspOutputBranch]]`, and `link_rtsp_output_branches(...) -> None`.
- Extends: `create_h264_encoder_elements(..., element_name_suffix: str = "")` and `create_output_elements(..., rtsp_location: str | None = None, element_name_suffix: str = "")`.

- [ ] **Step 1: Write failing branch-construction tests**

```python
class FakeElement:
    def __init__(self, factory, name):
        self.factory = factory
        self.name = name
        self.properties = {}

    def set_property(self, name, value):
        self.properties[name] = value


def test_create_rtsp_output_branches_uses_unique_names_and_locations():
    created = []

    def make_element(factory, name):
        element = FakeElement(factory, name)
        created.append(element)
        return element

    def create_output(camera_id, location):
        queue = make_element("queue", f"output-{camera_id}")
        sink = make_element("rtspclientsink", f"sink-{camera_id}")
        sink.set_property("location", location)
        return [queue, sink]

    demux, branches = create_rtsp_output_branches(
        source_entries=[
            (0, "camera_1", {}, "rtsp://input/1"),
            (1, "camera_2", {}, "rtsp://input/2"),
        ],
        locations={
            "camera_1": "rtsp://media:8554/camera_1",
            "camera_2": "rtsp://media:8554/camera_2",
        },
        make_element=make_element,
        create_output_elements=create_output,
    )

    assert demux.factory == "nvstreamdemux"
    assert [branch.camera_id for branch in branches] == ["camera_1", "camera_2"]
    assert branches[0].elements[-1].properties["location"].endswith("/camera_1")
    assert branches[1].elements[-1].properties["location"].endswith("/camera_2")
    assert len({element.name for element in created}) == len(created)
```

Add a separate test where fake `nvstreamdemux.get_request_pad("src_1")` links to the first element's static `sink` pad, and assert a missing pad or non-OK `PadLinkReturn` raises a `RuntimeError` containing `camera_2` and `pad_id=1`.

- [ ] **Step 2: Run the branch tests and verify RED**

Run: `rtk pytest tests/test_deepstream_rtsp_output.py tests/test_deepstream_processor.py -q -k 'rtsp_output_branch or unique_h264_element_names'`

Expected: FAIL because branch types/functions and suffixed encoder element names do not exist.

- [ ] **Step 3: Implement branch data and demux linking**

Add the following public shape to `_deepstream_rtsp_output.py`:

```python
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class RtspOutputBranch:
    camera_id: str
    pad_id: int
    elements: list[Any]


def create_rtsp_output_branches(
    *,
    source_entries: list[tuple[int, str, dict[str, Any], str]],
    locations: dict[str, str],
    make_element: Callable[[str, str], Any],
    create_output_elements: Callable[[str, str], list[Any]],
) -> tuple[Any, list[RtspOutputBranch]]:
    demux = make_element("nvstreamdemux", "output-demux")
    branches = [
        RtspOutputBranch(
            camera_id=camera_id,
            pad_id=pad_id,
            elements=create_output_elements(camera_id, locations[camera_id]),
        )
        for pad_id, camera_id, _info, _source_uri in source_entries
    ]
    return demux, branches
```

Implement `link_rtsp_output_branches` so it links each branch's elements in order, requests `src_<pad_id>` from the demux, obtains the first element's `sink` pad, compares the result with `gst_module.PadLinkReturn.OK`, and raises a camera-specific `RuntimeError` on every missing/failed pad case.

Update encoder/output creation so every GStreamer name gets `-<camera_id>` when `element_name_suffix` is non-empty. For RTSP calls, require `rtsp_location` and set it directly instead of reading `DS_RTSP_LOCATION` inside the builder. Insert a leaky `queue` as the first element of every camera branch and configure it with the existing output queue policy.

- [ ] **Step 4: Run branch and existing output tests and verify GREEN**

Run: `rtk pytest tests/test_deepstream_rtsp_output.py tests/test_deepstream_processor.py -q -k 'rtsp or output_elements or h264 or output_queue'`

Expected: all selected tests PASS, including existing MPEG-TS and POC-fix tests.

- [ ] **Step 5: Commit branch construction**

```bash
rtk git add src/core/_deepstream_rtsp_output.py src/core/_deepstream_pipeline_builder.py tests/test_deepstream_rtsp_output.py tests/test_deepstream_processor.py
rtk git commit -m "Add DeepStream camera output branches"
```

### Task 3: Integrate demux branches into DeepStreamProcessor

**Files:**
- Modify: `src/core/_deepstream_pipeline_builder.py`
- Modify: `src/core/deepstream_processor.py`
- Modify: `tests/test_deepstream_processor.py`

**Interfaces:**
- Consumes: `resolve_rtsp_locations`, `create_rtsp_output_branches`, `link_rtsp_output_branches`, and existing `source_entries`.
- Produces: an extended `DeepStreamPipelineElements` containing `output_demux: Any | None` and `rtsp_output_branches: list[RtspOutputBranch]`.
- Preserves: `_create_output_elements()` with default arguments for existing tests and non-RTSP modes.

- [ ] **Step 1: Write failing pipeline topology tests**

```python
def test_link_pipeline_routes_annotated_output_through_demux():
    osd = _FakeElement("osd")
    demux = _FakeElement("output-demux")
    probe = _FakeElement("primary-infer")
    probe.static_pad = _FakePad()
    elements = DeepStreamPipelineElements(
        streammux=_FakeElement("streammux"),
        nvinfer=probe,
        pphuman_infer=None,
        helmet_infer=None,
        tracker=None,
        converter=_FakeElement("converter"),
        osd=osd,
        tee=None,
        output_queue=None,
        preview_elements=[],
        output_elements=[],
        output_demux=demux,
        rtsp_output_branches=[],
    )

    link_deepstream_pipeline_path(
        elements,
        link_or_raise=DeepStreamProcessor._link_or_raise,
        gst_module=types.SimpleNamespace(PadProbeType=types.SimpleNamespace(BUFFER="buffer")),
        primary_probe_callback=MagicMock(),
        link_preview_branch=MagicMock(),
        link_rtsp_branches=MagicMock(),
        pphuman_gie_id=3,
        pphuman_infer_config="unused.txt",
    )

    assert demux in osd.linked_to
```

Also add focused helper-level assertions that `link_deepstream_pipeline_path` links `osd -> output-demux` when preview is disabled, and `osd -> tee -> output-queue -> output-demux` when preview is enabled. Retain an assertion that `fakesink` mode never creates `nvstreamdemux`.

- [ ] **Step 2: Run topology tests and verify RED**

Run: `rtk pytest tests/test_deepstream_processor.py -q -k 'per_source or output_demux or preview_branch'`

Expected: FAIL because the element bundle has no demux or RTSP branch fields.

- [ ] **Step 3: Extend the element bundle and processor assembly**

Extend the dataclass fields and element collection:

```python
output_demux: Optional[Any]
rtsp_output_branches: List[RtspOutputBranch]

# in all_elements()
if self.output_demux is not None:
    elements.append(self.output_demux)
for branch in self.rtsp_output_branches:
    elements.extend(branch.elements)
```

In `_build_pipeline`, resolve locations from the active source entry camera IDs using `os.environ.get("DS_RTSP_LOCATION_TEMPLATE")` and `os.environ.get("DS_RTSP_LOCATION")`. Only for `rtsp`, `rtsp_publish`, and `rtsp-publish`, create the demux and camera branches. For other modes, continue calling the existing single `_create_output_elements()` path.

Update linking in this exact order:

```python
previous = elements.osd
if elements.tee is not None and elements.output_queue is not None:
    previous = link_preview_branch(...)
if elements.output_demux is not None:
    link_or_raise(previous, elements.output_demux, "output path -> nvstreamdemux link 실패")
    link_rtsp_output_branches(...)
else:
    for element in elements.output_elements:
        link_or_raise(previous, element, None)
        previous = element
```

Keep probe registration on the existing inference/tracker probe element; do not move event processing onto output branch pads.

- [ ] **Step 4: Run processor tests and verify GREEN**

Run: `rtk pytest tests/test_deepstream_processor.py tests/test_deepstream_rtsp_output.py -q`

Expected: all tests PASS.

- [ ] **Step 5: Commit processor integration**

```bash
rtk git add src/core/_deepstream_pipeline_builder.py src/core/deepstream_processor.py tests/test_deepstream_processor.py
rtk git commit -m "Publish annotated streams per camera"
```

### Task 4: Wire dynamic publisher paths into Compose and MediaMTX

**Files:**
- Modify: `docker-compose.jetson.yml`
- Modify: `.env.jetson`
- Modify: `.env.jetson.example`
- Modify: `config/mediamtx.yml`
- Modify: `scripts/health/check_compose_runtime_assumptions.py`
- Modify: `tests/test_check_compose_runtime_assumptions.py`
- Modify: `docs/guides/DEPLOYMENT_ENVIRONMENT_VARIABLES.md`

**Interfaces:**
- Consumes: `DS_RTSP_LOCATION_TEMPLATE` and MediaMTX `all_others` path configuration.
- Produces: `check_per_camera_rtsp_wiring(...) -> dict[str, Any]`, included in `run_checks()`.

- [ ] **Step 1: Write a failing deployment wiring test**

```python
def test_per_camera_rtsp_wiring_requires_template_and_dynamic_mediamtx_path():
    result = runtime_checks.check_per_camera_rtsp_wiring(
        jetson_compose_text=(
            "DS_RTSP_LOCATION_TEMPLATE: "
            "${DS_RTSP_LOCATION_TEMPLATE:-rtsp://cctv-media-server:8554/{camera_id}}\n"
        ),
        mediamtx_text="paths:\n  all_others:\n    source: publisher\n",
        jetson_env_example_text=(
            "DS_RTSP_LOCATION_TEMPLATE=rtsp://cctv-media-server:8554/{camera_id}\n"
        ),
    )

    assert result["passed"] is True
```

Add three negative parameterized cases, each omitting one required entry, and assert `passed is False` with the missing file/setting in `detail`.

- [ ] **Step 2: Run the wiring test and verify RED**

Run: `rtk pytest tests/test_check_compose_runtime_assumptions.py -q -k per_camera_rtsp`

Expected: FAIL because `check_per_camera_rtsp_wiring` does not exist.

- [ ] **Step 3: Implement configuration and health checks**

Use this Compose default:

```yaml
DS_RTSP_LOCATION_TEMPLATE: ${DS_RTSP_LOCATION_TEMPLATE:-rtsp://cctv-media-server:8554/{camera_id}}
```

Use this MediaMTX fallback while retaining the explicit `sample_eval` entry:

```yaml
paths:
  sample_eval:
    source: publisher

  all_others:
    source: publisher
```

Replace the tracked `.env.jetson` single `DS_RTSP_LOCATION` assignment and add the same template to `.env.jetson.example`. Implement `check_per_camera_rtsp_wiring` with injectable text arguments like the existing wiring checks, verify all three exact configuration fragments, and include it in `run_checks()`.

Add the deployment environment variable table row:

```markdown
| `DS_RTSP_LOCATION_TEMPLATE` | `rtsp://cctv-media-server:8554/{camera_id}` | 동일 | 카메라 ID별 분석 영상 RTSP 게시 URL 템플릿 |
```

Document that WebRTC uses `http://<Jetson-IP>:8889/<camera_id>/` and legacy `DS_RTSP_LOCATION` is supported only for one active camera.

- [ ] **Step 4: Run configuration tests and Compose validation**

Run: `rtk pytest tests/test_check_compose_runtime_assumptions.py -q`

Expected: all tests PASS.

Run: `rtk proxy docker compose -f docker-compose.jetson.yml config --quiet`

Expected: exit code 0 with no Compose interpolation or YAML error.

Run: `rtk proxy .venv/bin/python scripts/health/check_compose_runtime_assumptions.py`

Expected: every check reports PASS, including per-camera RTSP wiring.

- [ ] **Step 5: Commit deployment wiring**

```bash
rtk git add docker-compose.jetson.yml .env.jetson .env.jetson.example config/mediamtx.yml scripts/health/check_compose_runtime_assumptions.py tests/test_check_compose_runtime_assumptions.py docs/guides/DEPLOYMENT_ENVIRONMENT_VARIABLES.md
rtk git commit -m "Configure dynamic camera media paths"
```

### Task 5: Run regression checks and prepare Jetson validation

**Files:**
- Modify only if a validation failure reveals a defect in files already listed above.

**Interfaces:**
- Consumes: completed implementation from Tasks 1-4.
- Produces: verified repository state and a concrete Jetson runtime validation command set.

- [ ] **Step 1: Run focused RTSP and deployment tests**

Run: `rtk pytest tests/test_deepstream_rtsp_output.py tests/test_deepstream_processor.py tests/test_check_compose_runtime_assumptions.py -q`

Expected: all tests PASS.

- [ ] **Step 2: Run Ruff**

Run: `rtk ruff check src/core/_deepstream_rtsp_output.py src/core/_deepstream_pipeline_builder.py src/core/deepstream_processor.py scripts/health/check_compose_runtime_assumptions.py tests/test_deepstream_rtsp_output.py tests/test_deepstream_processor.py tests/test_check_compose_runtime_assumptions.py`

Expected: no lint errors.

- [ ] **Step 3: Run the complete unit test suite**

Run: `rtk pytest tests/ -q`

Expected: all tests PASS; environment-specific skips are reported separately.

- [ ] **Step 4: Validate on Jetson with two configured cameras**

Run after the updated services are deployed:

```bash
rtk docker compose -f docker-compose.jetson.yml up -d --build cctv-media-server cctv-ai-engine
rtk docker logs cctv-ai-engine --tail 200
rtk proxy ffprobe -v error -rtsp_transport tcp -show_entries stream=codec_name,width,height -of json rtsp://127.0.0.1:8554/camera_1
rtk proxy ffprobe -v error -rtsp_transport tcp -show_entries stream=codec_name,width,height -of json rtsp://127.0.0.1:8554/camera_2
```

Expected: engine logs contain no demux pad or RTSP sink errors; both `ffprobe` calls report an H.264 video stream at the configured output dimensions. Visually open `http://<Jetson-IP>:8889/camera_1/` and `http://<Jetson-IP>:8889/camera_2/` and confirm each path shows the correct camera with AI overlays.

- [ ] **Step 5: Record final repository status**

Run: `rtk git status --short --branch`

Expected: the implementation branch is ahead by the planned commits and the worktree is clean. Do not claim Jetson multi-camera runtime verification unless Step 4 was executed on hardware with two live sources.

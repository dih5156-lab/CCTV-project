"""GIE별 출력 메타 관측. 캐시 결과를 신규 추론 완료로 추정하지 않는다."""

from __future__ import annotations

import time
from threading import Lock
from typing import Any

FRAME_STATES = (
    "tensor_output",
    "no_objects",
    "classifier_only",
    "unobserved",
    "unknown",
)
ELEMENT_PROPERTIES = (
    "unique-id",
    "process-mode",
    "batch-size",
    "interval",
    "output-tensor-meta",
)


def _metadata_items(node: Any, cast: Any):
    while node is not None:
        yield cast(node.data)
        try:
            node = node.next
        except StopIteration:
            return


def _has_tensor(node: Any, bindings: Any, component_id: int) -> bool:
    found = False
    for meta in _metadata_items(node, bindings.NvDsUserMeta.cast):
        if (
            meta.base_meta.meta_type
            != bindings.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META
        ):
            continue
        tensor = bindings.NvDsInferTensorMeta.cast(meta.user_meta_data)
        if type(tensor.unique_id) is not int or tensor.unique_id < 0:
            raise ValueError("invalid tensor GIE ID")
        found = found or tensor.unique_id == component_id
    return found


def observe_model_frame(
    frame: Any, bindings: Any, properties: dict[str, Any]
) -> tuple[str, int, int]:
    """서로 배타적인 프레임 상태와 tensor/classifier-only 객체 수를 반환한다.

    no_objects는 SGIE 경계의 객체 목록이 빈 경우다. 객체가 있어도 클래스·크기
    필터를 통과한 ROI라는 보장은 없다. classifier_only의 캐시/비동기 여부도
    이 메타만으로 확정할 수 없으므로 신규 완료로 세지 않는다.
    """
    component_id = properties.get("unique-id")
    mode = properties.get("process-mode")
    if (
        type(component_id) is not int
        or component_id < 0
        or type(mode) is not int
        or mode not in (1, 2)
        or properties.get("output-tensor-meta") is not True
    ):
        return "unknown", 0, 0
    try:
        if mode == 1:
            found = _has_tensor(frame.frame_user_meta_list, bindings, component_id)
            return ("tensor_output" if found else "unobserved"), 0, 0
        objects = tensors = classifiers = 0
        for obj in _metadata_items(frame.obj_meta_list, bindings.NvDsObjectMeta.cast):
            objects += 1
            if _has_tensor(obj.obj_user_meta_list, bindings, component_id):
                tensors += 1
            elif any(
                meta.unique_component_id == component_id
                for meta in _metadata_items(
                    obj.classifier_meta_list, bindings.NvDsClassifierMeta.cast
                )
            ):
                classifiers += 1
        if not objects:
            return "no_objects", 0, 0
        if tensors:
            return "tensor_output", tensors, classifiers
        if classifiers:
            return "classifier_only", 0, classifiers
        return "unobserved", 0, 0
    except Exception:
        return "unknown", 0, 0


class ModelInferenceMetrics:
    """최종 metadata queue에서 관측하며 pipeline 재생성 때 세대를 갱신한다."""

    def __init__(self, *, clock=time.monotonic):
        self._clock = clock
        self._lock = Lock()
        self._generation = 0
        self._models: dict[str, Any] = {}
        self._metadata_errors = 0

    def reset(self, camera_ids, elements):
        camera_ids = tuple(camera_ids)
        models = {}
        for name, element in elements.items():
            properties = {}
            for key in ELEMENT_PROPERTIES:
                try:
                    value = element.get_property(key)
                    # GObject enums are int subclasses; never expose native objects.
                    properties[key] = (
                        value
                        if type(value) is bool
                        else int(value) if isinstance(value, int) else None
                    )
                except Exception:
                    properties[key] = None
            models[name] = {
                "enabled": element is not None,
                "properties": properties,
                "property_observed_at": "pipeline_build",
                "cameras": {
                    camera: {
                        "frames_seen": 0,
                        **{f"frames_{state}": 0 for state in FRAME_STATES},
                        "objects_with_tensor": 0,
                        "objects_classifier_only": 0,
                        "last_frame_at": None,
                        "last_tensor_at": None,
                    }
                    for camera in camera_ids
                },
            }
        ids = [m["properties"]["unique-id"] for m in models.values() if m["enabled"]]
        for model in models.values():
            model["unique_id_conflict"] = (
                model["enabled"] and ids.count(model["properties"]["unique-id"]) > 1
            )
        with self._lock:
            self._generation += 1
            self._models = models
            self._metadata_errors = 0

    def record_batch(self, batch_meta, *, pyds_module, pad_to_camera):
        now = self._clock()
        with self._lock:
            for data in _metadata_items(
                batch_meta.frame_meta_list, lambda value: value
            ):
                try:
                    frame = pyds_module.NvDsFrameMeta.cast(data)
                    camera = pad_to_camera.get(frame.source_id)
                except Exception:
                    self._metadata_errors += 1
                    continue
                for model in self._models.values():
                    counts = model["cameras"].get(camera)
                    if not model["enabled"] or counts is None:
                        continue
                    state, tensors, classifiers = (
                        ("unknown", 0, 0)
                        if model["unique_id_conflict"]
                        else observe_model_frame(
                            frame, pyds_module, model["properties"]
                        )
                    )
                    counts["frames_seen"] += 1
                    counts[f"frames_{state}"] += 1
                    counts["objects_with_tensor"] += tensors
                    counts["objects_classifier_only"] += classifiers
                    counts["last_frame_at"] = now
                    if state == "tensor_output":
                        counts["last_tensor_at"] = now

    def snapshot(self):
        with self._lock:
            now = self._clock()
            models = {}
            for name, model in self._models.items():
                cameras = {}
                for camera, counts in model["cameras"].items():
                    values = dict(counts)
                    for kind in ("frame", "tensor"):
                        last = values.pop(f"last_{kind}_at")
                        values[f"last_{kind}_age_seconds"] = (
                            max(0, now - last) if last is not None else None
                        )
                    cameras[camera] = values
                models[name] = {
                    **model,
                    "properties": dict(model["properties"]),
                    "cameras": cameras,
                }
            return {
                "pipeline_generation": self._generation,
                "metadata_errors": self._metadata_errors,
                "models": models,
            }

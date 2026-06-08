"""외형 검색 상태 진단 서비스."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Generator, List, Optional, Set

from pydantic import BaseModel, Field

from ..config.config import PROJECT_ROOT
from ..storage import SQLiteDatabase

_BAG_LABEL_ALIASES = {
    "backpack",
    "back_pack",
    "rucksack",
    "handbag",
    "hand_bag",
    "purse",
    "suitcase",
    "luggage",
    "travel_bag",
    "carry_on",
}


class AppearanceFieldStatus(BaseModel):
    """속성 필드별 준비 상태."""

    field: str = Field(description="대시보드/검색에서 사용하는 필드 이름")
    enabled: bool = Field(description="설정상 이 필드가 활성화되어 있는지 여부")
    ready: bool = Field(description="현재 런타임/모델/라벨 기준으로 실제 적재가 가능한 상태인지 여부")
    source: str = Field(description="값 생성 경로. 예: face_recognition, helmet_detection, yolo_nearby_objects")
    observed_count: int = Field(default=0, description="현재 DB에서 실제로 채워진 건수")
    observed_ratio: float = Field(default=0.0, description="total_records 대비 실제 관측 비율 (0.0~1.0)")
    reason: Optional[str] = Field(default=None, description="ready=false 인 경우 주된 원인 설명")


class AppearanceDataStats(BaseModel):
    """외형 로그 적재 통계."""

    total_records: int = Field(description="appearance_log 총 레코드 수")
    gender_filled: int = Field(description="gender 값이 채워진 레코드 수")
    helmet_true: int = Field(description="has_helmet=1 레코드 수")
    backpack_true: int = Field(description="has_backpack=1 레코드 수")
    handbag_true: int = Field(description="has_handbag=1 레코드 수")
    suitcase_true: int = Field(description="has_suitcase=1 레코드 수")
    latest_timestamp: Optional[float] = Field(default=None, description="가장 최근 appearance_log timestamp")


class AppearanceRuntimeStatus(BaseModel):
    """대시보드용 외형 검색 준비 상태."""

    db_path: str = Field(description="현재 조회 중인 외형 로그 DB 경로")
    backend: str = Field(description="현재 외형 속성 백엔드 이름. 예: hsv, pphuman")
    fields: List[AppearanceFieldStatus] = Field(description="필드별 활성화/준비/관측 상태")
    data_stats: AppearanceDataStats = Field(description="DB 기준 누적 적재 통계")
    backend_counts: Dict[str, int] = Field(description="attribute_backend별 적재 건수. 과거 데이터는 unknown 으로 집계")
    warnings: List[str] = Field(description="운영/대시보드에서 바로 보여줄 진단 메시지")
    next_steps: List[str] = Field(description="다음 확인/조치 권장 사항")


def build_runtime_status(db_path: Path) -> AppearanceRuntimeStatus:
    """외형 검색 상태 응답을 구성한다."""
    data_stats = _collect_data_stats(db_path)
    backend = os.environ.get("APPEARANCE_BACKEND", "hsv").strip().lower()
    fields = _build_field_statuses(data_stats)
    backend_counts = _collect_backend_counts(db_path)
    warnings = _build_runtime_warnings(
        backend=backend,
        fields=fields,
        data_stats=data_stats,
        backend_counts=backend_counts,
    )
    return AppearanceRuntimeStatus(
        db_path=str(db_path),
        backend=backend,
        fields=fields,
        data_stats=data_stats,
        backend_counts=backend_counts,
        warnings=warnings,
        next_steps=_build_next_steps(fields=fields, warnings=warnings),
    )


def _csv_env(name: str, default: str = "") -> List[str]:
    raw = os.environ.get(name, default)
    return [
        item.strip().lower()
        for item in raw.split(",")
        if item and item.strip()
    ]


def _truthy_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_project_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _load_active_camera_flags() -> Optional[List[Dict[str, bool]]]:
    """활성 카메라의 모델 on/off 설정을 읽는다.

    상태 API는 public API 컨테이너에서 실행될 수 있으므로 CAMERAS_JSON을 우선하고,
    없으면 프로젝트 기본 cameras.json을 사용한다.
    """
    cameras_path = _resolve_project_path(os.environ.get("CAMERAS_JSON", "cameras.json"))
    if not cameras_path.exists():
        return None
    try:
        cameras = json.loads(cameras_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(cameras, list):
        return None

    active_flags: List[Dict[str, bool]] = []
    detection_aliases = {
        "use_helmet": "helmet",
        "use_face": "face",
        "use_appearance": "appearance",
    }
    for camera in cameras:
        if not isinstance(camera, dict) or camera.get("enabled") is False:
            continue
        model_settings = camera.get("model_settings")
        detections = camera.get("detections")
        detection_set = {
            str(item).strip().lower()
            for item in detections
            if isinstance(item, str)
        } if isinstance(detections, list) else set()
        flags: Dict[str, bool] = {}
        for flag_name, detection_name in detection_aliases.items():
            if isinstance(model_settings, dict) and flag_name in model_settings:
                flags[flag_name] = bool(model_settings[flag_name])
            elif detection_name in detection_set:
                flags[flag_name] = True
        active_flags.append(flags)
    return active_flags


def _camera_flag_enabled(flag_name: str) -> Optional[bool]:
    active_flags = _load_active_camera_flags()
    if active_flags is None:
        return None
    if not active_flags:
        return False
    explicit_values = [
        flags[flag_name]
        for flags in active_flags
        if flag_name in flags
    ]
    if not explicit_values:
        return None
    return any(explicit_values)


def _env_and_camera_flag(env_name: str, flag_name: str, default: bool) -> bool:
    env_enabled = _truthy_env(env_name, default)
    camera_enabled = _camera_flag_enabled(flag_name)
    if camera_enabled is None:
        return env_enabled
    return env_enabled and camera_enabled


def _load_attribute_label_fields() -> Optional[Set[str]]:
    label_map_path = os.environ.get("APPEARANCE_LABEL_MAP_PATH", "").strip()
    if not label_map_path:
        return None
    path = _resolve_project_path(label_map_path)
    try:
        label_map = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    labels = label_map.get("labels")
    if not isinstance(labels, list):
        return None
    return {
        str(entry.get("field", "")).strip()
        for entry in labels
        if isinstance(entry, dict) and str(entry.get("field", "")).strip()
    }


@contextmanager
def _connect(db_path: Path) -> Generator[sqlite3.Connection, None, None]:
    """SQLite 연결을 열고 켄텍스트 종료 시 반드시 닫는다."""
    conn = SQLiteDatabase(db_path).connect()
    try:
        yield conn
    finally:
        conn.close()


def _collect_data_stats(db_path: Path) -> AppearanceDataStats:
    try:
        with _connect(db_path) as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) AS total_records,
                    SUM(CASE WHEN gender IS NOT NULL AND gender != '' THEN 1 ELSE 0 END) AS gender_filled,
                    SUM(CASE WHEN has_helmet = 1 THEN 1 ELSE 0 END) AS helmet_true,
                    SUM(CASE WHEN has_backpack = 1 THEN 1 ELSE 0 END) AS backpack_true,
                    SUM(CASE WHEN has_handbag = 1 THEN 1 ELSE 0 END) AS handbag_true,
                    SUM(CASE WHEN has_suitcase = 1 THEN 1 ELSE 0 END) AS suitcase_true,
                    MAX(timestamp) AS latest_timestamp
                FROM appearance_log
                """
            ).fetchone()
    except sqlite3.Error:
        row = {
            "total_records": 0,
            "gender_filled": 0,
            "helmet_true": 0,
            "backpack_true": 0,
            "handbag_true": 0,
            "suitcase_true": 0,
            "latest_timestamp": None,
        }

    return AppearanceDataStats(
        total_records=int(row["total_records"] or 0),
        gender_filled=int(row["gender_filled"] or 0),
        helmet_true=int(row["helmet_true"] or 0),
        backpack_true=int(row["backpack_true"] or 0),
        handbag_true=int(row["handbag_true"] or 0),
        suitcase_true=int(row["suitcase_true"] or 0),
        latest_timestamp=float(row["latest_timestamp"]) if row["latest_timestamp"] is not None else None,
    )


def _collect_backend_counts(db_path: Path) -> Dict[str, int]:
    try:
        with _connect(db_path) as conn:
            rows = conn.execute(
                """
                SELECT COALESCE(NULLIF(attribute_backend, ''), 'unknown') AS backend_name,
                       COUNT(*) AS count
                FROM appearance_log
                GROUP BY COALESCE(NULLIF(attribute_backend, ''), 'unknown')
                ORDER BY count DESC
                """
            ).fetchall()
    except sqlite3.Error:
        return {}

    return {
        str(row["backend_name"]): int(row["count"] or 0)
        for row in rows
    }


def _build_field_statuses(data_stats: AppearanceDataStats) -> List[AppearanceFieldStatus]:
    backend = os.environ.get("APPEARANCE_BACKEND", "hsv").strip().lower()
    face_enabled = _env_and_camera_flag("DS_FACE_ENABLED", "use_face", False)
    appearance_runtime_enabled = _truthy_env("DS_APPEARANCE_ENABLED", False) or _truthy_env("APPEARANCE_ENABLED", False)
    camera_appearance_enabled = _camera_flag_enabled("use_appearance")
    appearance_enabled = (
        appearance_runtime_enabled
        if camera_appearance_enabled is None
        else appearance_runtime_enabled and camera_appearance_enabled
    )
    helmet_enabled = _env_and_camera_flag("DS_HELMET_ENABLED", "use_helmet", True)
    yolo_labels = set(_csv_env("DS_YOLO_LABELS", "person"))
    bag_labels = sorted(label for label in yolo_labels if label in _BAG_LABEL_ALIASES)
    attribute_label_fields = _load_attribute_label_fields()
    bag_model_ready = bool(bag_labels) or backend != "hsv"
    total_records = max(data_stats.total_records, 1)

    return [
        AppearanceFieldStatus(
            field="gender",
            enabled=face_enabled,
            ready=face_enabled,
            source="face_recognition",
            observed_count=data_stats.gender_filled,
            observed_ratio=round(data_stats.gender_filled / total_records, 4),
            reason=None if face_enabled else "DS_FACE_ENABLED 또는 얼굴 인식 경로가 비활성화됨",
        ),
        AppearanceFieldStatus(
            field="has_helmet",
            enabled=appearance_enabled and helmet_enabled,
            ready=appearance_enabled and helmet_enabled,
            source="helmet_detection",
            observed_count=data_stats.helmet_true,
            observed_ratio=round(data_stats.helmet_true / total_records, 4),
            reason=(
                None
                if appearance_enabled and helmet_enabled
                else "DS_APPEARANCE_ENABLED 또는 DS_HELMET_ENABLED가 비활성화됨"
            ),
        ),
        AppearanceFieldStatus(
            field="helmet_color",
            enabled=appearance_enabled and helmet_enabled,
            ready=appearance_enabled and helmet_enabled,
            source="helmet_detection+hsv",
            observed_count=data_stats.helmet_true,
            observed_ratio=round(data_stats.helmet_true / total_records, 4),
            reason=(
                None
                if appearance_enabled and helmet_enabled
                else "헬멧 감지 또는 외형 분석이 비활성화됨"
            ),
        ),
        *_build_bag_field_statuses(
            appearance_enabled=appearance_enabled,
            backend=backend,
            bag_model_ready=bag_model_ready,
            bag_labels=bag_labels,
            attribute_label_fields=attribute_label_fields,
            data_stats=data_stats,
            total_records=total_records,
        ),
    ]


def _build_bag_field_statuses(
    *,
    appearance_enabled: bool,
    backend: str,
    bag_model_ready: bool,
    bag_labels: List[str],
    attribute_label_fields: Optional[Set[str]],
    data_stats: AppearanceDataStats,
    total_records: int,
) -> List[AppearanceFieldStatus]:
    source = "attribute_backend" if backend != "hsv" else "yolo_nearby_objects"
    values = [
        ("has_backpack", data_stats.backpack_true),
        ("has_handbag", data_stats.handbag_true),
        ("has_suitcase", data_stats.suitcase_true),
    ]
    statuses: List[AppearanceFieldStatus] = []
    for name, count in values:
        field_ready = bag_model_ready
        if backend != "hsv" and attribute_label_fields is not None:
            field_ready = name in attribute_label_fields
        reason = None if appearance_enabled and field_ready else (
            f"backend={backend}, bag_labels={','.join(bag_labels) if bag_labels else 'none'}"
        )
        statuses.append(AppearanceFieldStatus(
            field=name,
            enabled=appearance_enabled,
            ready=appearance_enabled and field_ready,
            source=source,
            observed_count=count,
            observed_ratio=round(count / total_records, 4),
            reason=reason,
        ))
    return statuses


def _build_runtime_warnings(
    *,
    backend: str,
    fields: List[AppearanceFieldStatus],
    data_stats: AppearanceDataStats,
    backend_counts: Dict[str, int],
) -> List[str]:
    warnings: List[str] = []
    field_map = {field.field: field for field in fields}

    if data_stats.total_records == 0:
        warnings.append("appearance_log 데이터가 아직 없습니다. 재시작 후 새 이벤트가 적재되는지 먼저 확인하세요.")
        return warnings

    if backend_counts and set(backend_counts.keys()) == {"unknown"}:
        warnings.append("현재 DB는 과거 적재분 위주라 attribute_backend가 모두 unknown 입니다. 재시작 후 새 데이터부터 hsv/pphuman 값이 보이기 시작해야 정상입니다.")

    gender_field = field_map.get("gender")
    if gender_field and gender_field.ready and gender_field.observed_count == 0:
        warnings.append("gender는 설정상 활성화되어 있지만 실제 적재 건수가 0입니다. 얼굴 인식 경로(use_face/DS_FACE_ENABLED)와 얼굴 ROI 품질을 확인하세요.")

    helmet_field = field_map.get("has_helmet")
    if helmet_field and helmet_field.ready and helmet_field.observed_count == 0:
        warnings.append("has_helmet는 설정상 활성화되어 있지만 실제 적재 건수가 0입니다. 헬멧 이벤트가 생성되는지와 사람 머리 근처로 연결되는지 확인이 필요합니다.")

    bag_fields = [field_map.get("has_backpack"), field_map.get("has_handbag"), field_map.get("has_suitcase")]
    if all(field is not None and field.ready for field in bag_fields) and all(field.observed_count == 0 for field in bag_fields if field is not None):
        warnings.append("가방 계열 필드는 설정상 준비 상태지만 실제 적재가 0입니다. detector가 bag label을 실제로 내는지, 또는 속성 모델(pphuman)이 연결됐는지 확인하세요.")

    if backend == "hsv" and all((field_map.get(name).observed_count if field_map.get(name) else 0) == 0 for name in ("has_backpack", "has_handbag", "has_suitcase")):
        warnings.append("backend=hsv 환경에서는 bag 값이 detector nearby_objects에 의존합니다. DS_YOLO_LABELS 또는 person 모델이 bag 클래스를 내지 않으면 계속 0으로 남을 수 있습니다.")

    return warnings


def _build_next_steps(
    *,
    fields: List[AppearanceFieldStatus],
    warnings: List[str],
) -> List[str]:
    next_steps: List[str] = []
    field_map = {field.field: field for field in fields}

    if warnings:
        next_steps.append("AI 엔진 재시작 후 /api/v1/appearances/status 를 다시 조회해 backend_counts와 observed_count 변화를 확인하세요.")

    helmet_field = field_map.get("has_helmet")
    if helmet_field and helmet_field.ready:
        next_steps.append("has_helmet가 계속 0이면 헬멧 이벤트 로그와 사람 머리 bbox 정합 여부를 먼저 확인하세요.")

    bag_field = field_map.get("has_backpack")
    if bag_field and bag_field.ready:
        next_steps.append("bag 값이 계속 0이면 detector 출력 라벨(예: backpack/back_pack/luggage) 또는 pphuman 연결 여부를 확인하세요.")

    gender_field = field_map.get("gender")
    if gender_field and gender_field.ready:
        next_steps.append("gender 비율이 낮으면 얼굴 인식 사용 여부와 얼굴 crop 크기/가림 상태를 확인하세요.")

    deduped: List[str] = []
    for item in next_steps:
        if item not in deduped:
            deduped.append(item)
    return deduped

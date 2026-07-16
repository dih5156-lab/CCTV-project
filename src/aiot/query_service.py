from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

from src.aiot.contracts import AiQueryRequest


class LiveMatchProvider(Protocol):
    def search(
        self, filters: Mapping[str, Any], camera_ids: Sequence[str], limit: int
    ) -> list[Mapping[str, Any]]: ...


class AiQueryService:
    def __init__(self, appearance_log: Any, live_provider: LiveMatchProvider):
        self.appearance_log = appearance_log
        self.live_provider = live_provider

    def search(self, request: AiQueryRequest) -> list[dict[str, Any]]:
        rows: list[Mapping[str, Any]] = []
        if request.search_mode in {"history", "both"}:
            rows.extend(self._search_history(request))
        if request.search_mode in {"live", "both"}:
            rows.extend(
                self.live_provider.search(
                    request.filters, request.camera_ids, request.limit
                )
            )

        matches: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in rows:
            match = self._project(row)
            match_id = match["match_id"]
            if match_id in seen:
                continue
            seen.add(match_id)
            matches.append(match)
            if len(matches) >= request.limit:
                break
        return matches

    def _search_history(self, request: AiQueryRequest) -> list[Mapping[str, Any]]:
        cameras = request.camera_ids
        camera_id = None if cameras == ("*",) else cameras[0]
        filters = request.filters
        return self.appearance_log.search(
            camera_id=camera_id,
            upper_color=filters.get("upper_color"),
            lower_color=filters.get("lower_color"),
            has_helmet=filters.get("has_helmet"),
            helmet_color=filters.get("helmet_color"),
            has_backpack=filters.get("has_backpack"),
            has_handbag=filters.get("has_handbag"),
            has_suitcase=filters.get("has_suitcase"),
            gender=filters.get("gender"),
            age_group=filters.get("age_group"),
            face_name=filters.get("face_name"),
            time_from=request.time_from,
            time_to=request.time_to,
            limit=request.limit,
        )

    @staticmethod
    def _project(row: Mapping[str, Any]) -> dict[str, Any]:
        match_id = str(row.get("event_id") or row.get("match_id") or row.get("id"))
        metadata = row.get("attribute_metadata")
        confidence = metadata.get("confidence") if isinstance(metadata, Mapping) else None
        attribute_keys = (
            "upper_color",
            "lower_color",
            "has_helmet",
            "helmet_color",
            "has_backpack",
            "has_handbag",
            "has_suitcase",
            "gender",
            "age_group",
            "face_name",
        )
        return {
            "match_id": match_id,
            "camera_id": row.get("camera_id"),
            "occurred_at": row.get("timestamp") or row.get("occurred_at"),
            "confidence": confidence if confidence is not None else row.get("confidence"),
            "attributes": {
                key: row.get(key) for key in attribute_keys if row.get(key) is not None
            },
            "media_available": bool(row.get("crop_path") or row.get("media_available")),
        }


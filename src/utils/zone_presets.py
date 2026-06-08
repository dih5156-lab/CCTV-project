"""zone_presets.py - 위험구역 프리셋 영속 저장소.

그린 구역을 이름 붙여 파일에 저장하고 목록을 조회·삭제한다.
저장된 프리셋은 드롭박스 UI 등에서 꺼내 특정 카메라에 바로 적용할 수 있다.

저장 형식 (zone_presets.json)::

    [
      {
        "id": "a1b2c3d4",
        "name": "전기설비 구역",
        "zones": [
          {"id": "zone_1", "name": "전기설비", "polygon": [[100,100], ...]}
        ],
        "created_at": "2026-03-05T12:00:00"
      }
    ]
"""

import json
import logging
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from ..time_utils import now_kst

logger = logging.getLogger(__name__)


class ZonePresetStore:
    """위험구역 프리셋을 JSON 파일에 저장·조회·삭제하는 저장소."""

    def __init__(self, presets_path: str = "zone_presets.json") -> None:
        self.presets_path = Path(presets_path)

    # ------------------------------------------------------------------
    # 내부 I/O
    # ------------------------------------------------------------------

    def _load(self) -> List[Dict]:
        try:
            return json.loads(self.presets_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        except json.JSONDecodeError as exc:
            logger.error("zone_presets.json 파싱 오류: %s", exc)
            return []

    def _write(self, presets: List[Dict]) -> None:
        """atomic write — tmp 파일로 먼저 쓴 뒤 교체한다."""
        self.presets_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.presets_path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(presets, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(self.presets_path)

    # ------------------------------------------------------------------
    # 공개 API
    # ------------------------------------------------------------------

    def list_all(self) -> List[Dict]:
        """저장된 전체 프리셋 목록을 반환한다."""
        return self._load()

    def get(self, preset_id: str) -> Optional[Dict]:
        """ID로 특정 프리셋을 찾아 반환한다. 없으면 None."""
        return next((p for p in self._load() if p["id"] == preset_id), None)

    def save(self, name: str, zones: List[Dict]) -> Dict:
        """새 프리셋을 저장하고 저장된 프리셋 객체를 반환한다.

        매개변수:
            name: 프리셋 이름 (드롭박스에서 표시될 이름)
            zones: 구역 정의 리스트 [{'id': ..., 'name': ..., 'polygon': [...]}]

        반환값:
            {'id': ..., 'name': ..., 'zones': [...], 'created_at': ...}
        """
        presets = self._load()
        preset = self._new_preset(name, zones)
        presets.append(preset)
        self._write(presets)
        logger.info("프리셋 저장: %s (%s), zones=%d", preset["id"], name, len(zones))
        return preset

    def delete(self, preset_id: str) -> bool:
        """ID로 프리셋을 삭제한다. 삭제 성공이면 True, 없으면 False."""
        presets = self._load()
        new_presets = [p for p in presets if p["id"] != preset_id]
        if len(new_presets) == len(presets):
            return False
        self._write(new_presets)
        logger.info("프리셋 삭제: %s", preset_id)
        return True

    @staticmethod
    def _new_preset(name: str, zones: List[Dict]) -> Dict:
        return {
            "id": uuid.uuid4().hex[:8],
            "name": name,
            "zones": zones,
            "created_at": now_kst().isoformat(timespec="seconds"),
        }


__all__ = ["ZonePresetStore"]

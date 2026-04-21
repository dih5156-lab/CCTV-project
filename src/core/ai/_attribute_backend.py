"""외형 속성 모델 백엔드 인터페이스."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Protocol

import numpy as np


@dataclass(frozen=True)
class AttributeCrop:
    """속성 모델 입력용 사람 crop 정보."""

    frame: np.ndarray
    x: int
    y: int
    width: int
    height: int


class AttributeBackend(Protocol):
    """외형 속성 모델 백엔드 프로토콜."""

    backend_name: str

    def predict(self, crop: AttributeCrop) -> Dict[str, object]:
        """사람 crop에서 속성 예측 결과를 반환한다."""


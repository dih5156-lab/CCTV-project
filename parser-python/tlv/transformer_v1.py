"""
tlv/transformer_v1.py
=====================
Go 원본: aiot-tlv-parser/pkg/tlv/transformer_v1.go

구버전(V1) TLV 트랜스포머 모듈입니다.
구버전 펌웨어에서 전송하는 TLV 데이터의 ID 배치가 신버전과 다릅니다.
특히 created_at 필드의 타임스탬프 처리(Unix초 × 1000 = ms) 가 포함됩니다.
"""

from typing import Any, Dict


class TransformerV1:
    """
    구버전(V1) TLV 데이터 변환기
    Go: type TransformerV1 struct{}

    V1은 테이블 3(디바이스 정보)를 지원하지 않습니다.
    created_at 필드가 V0에 없고 V1에서는 Unix초 타임스탬프로 포함됩니다.
    (× 1000 하여 밀리초로 저장)
    """

    def transform(self, table_name: int, data: Dict[str, Any], tlv_items: list) -> Dict[str, Any]:
        """
        테이블 ID에 따라 V1 파싱 메서드 호출
        Go: func (t *TransformerV1) Transform(tableName int, data map[string]interface{}, tlvItems []TLVItem) (map[string]interface{}, error)
        """
        handlers = {
            34950: self._parse34950,
            34952: self._parse34952,
            34954: self._parse34954,
            34955: self._parse34955,
            34956: self._parse34956,
            34957: self._parse34957,
            34958: self._parse34958,
        }
        handler = handlers.get(table_name)
        if handler is None:
            raise ValueError(f"unsupported table name for v1: {table_name}")
        return handler(data, tlv_items)

    def _parse34950(self, data: Dict[str, Any], tlv_items: list) -> Dict[str, Any]:
        """
        Object 34950 V1 - 하천 모니터링 (V1은 데이터 그대로 반환)
        Go: func (t *TransformerV1) parse34950(...) → data 그대로 반환
        """
        # V1에서 34950은 변환 없이 그대로 반환
        return data

    def _parse34952(self, data: Dict[str, Any], tlv_items: list) -> Dict[str, Any]:
        """
        Object 34952 V1 - 침수 감지 (V1은 데이터 그대로 반환)
        Go: func (t *TransformerV1) parse34952(...) → data 그대로 반환
        """
        return data

    def _parse34954(self, data: Dict[str, Any], tlv_items: list) -> Dict[str, Any]:
        """
        Object 34954 V1 - 온습도 파싱
        Go: func (t *TransformerV1) parse34954(...)

        V1 ID → 필드 매핑 (V0와 다름):
          1 → temperature      (온도)
          2 → reporting_period (보고 주기)
          3 → (주석 처리됨, 미사용)
          4 → created_at       (Unix초 × 1000 = 밀리초)
        """
        for tlv in tlv_items:
            if tlv.id == 1:
                data["temperature_c"] = tlv.value
            elif tlv.id == 2:
                data["reporting_period_s"] = tlv.value
            elif tlv.id == 4:
                # Go: if val, ok := tlv.Value.(int64); ok { data["created_at"] = val * 1000 }
                if isinstance(tlv.value, int):
                    data["created_at"] = tlv.value * 1000
        return data

    def _parse34955(self, data: Dict[str, Any], tlv_items: list) -> Dict[str, Any]:
        """
        Object 34955 V1 - 경사계 파싱
        Go: func (t *TransformerV1) parse34955(...)

        V1 ID → 필드 매핑:
          1 → angle_x                   (X축 각도)
          2 → angle_y                   (Y축 각도)
          3 → reporting_period          (보고 주기)
          4 → (주석 처리됨, 미사용)
          5 → reporting_angle_threshold  (각도 임계값)
          7 → created_at               (Unix초 × 1000)
        """
        for tlv in tlv_items:
            if tlv.id == 1:
                data["angle_x_deg"] = tlv.value
            elif tlv.id == 2:
                data["angle_y_deg"] = tlv.value
            elif tlv.id == 3:
                data["reporting_period_s"] = tlv.value
            elif tlv.id == 5:
                data["reporting_angle_threshold_deg"] = tlv.value
            elif tlv.id == 7:
                if isinstance(tlv.value, int):
                    data["created_at"] = tlv.value * 1000
        return data

    def _parse34956(self, data: Dict[str, Any], tlv_items: list) -> Dict[str, Any]:
        """
        Object 34956 V1 - 화재 경보 파싱
        Go: func (t *TransformerV1) parse34956(...)

        V1 ID → 필드 매핑:
          0     → fire_alarm       (화재 감지)
          5518  → created_at       (타임스탬프)
          26241 → reporting_period (ms → 초)
        """
        for tlv in tlv_items:
            if tlv.id == 0:
                data["fire_alarm"] = tlv.value
            elif tlv.id == 5518:
                data["created_at"] = tlv.value
            elif tlv.id == 26241:
                # Go: float64(val) / 1000.0
                if isinstance(tlv.value, int):
                    data["reporting_period_s"] = tlv.value / 1000.0
        return data

    def _parse34957(self, data: Dict[str, Any], tlv_items: list) -> Dict[str, Any]:
        """
        Object 34957 V1 - 복합 요약1 파싱
        Go: func (t *TransformerV1) parse34957(...)

        V1 ID → 필드 매핑:
          1 → temperature (온도)
          2 → angle_x    (X축 각도)
          3 → angle_y    (Y축 각도)
          4 → created_at (Unix초 × 1000)
        """
        for tlv in tlv_items:
            if tlv.id == 1:
                data["temperature_c"] = tlv.value
            elif tlv.id == 2:
                data["angle_x_deg"] = tlv.value
            elif tlv.id == 3:
                data["angle_y_deg"] = tlv.value
            elif tlv.id == 4:
                if isinstance(tlv.value, int):
                    data["created_at"] = tlv.value * 1000
        return data

    def _parse34958(self, data: Dict[str, Any], tlv_items: list) -> Dict[str, Any]:
        """
        Object 34958 V1 - 복합 요약2 파싱
        Go: func (t *TransformerV1) parse34958(...)

        V1 ID → 필드 매핑:
          1  → acc_x      (X축 가속도)
          2  → acc_y      (Y축 가속도)
          3  → acc_z      (Z축 가속도)
          4  → gyro_x     (X축 자이로)
          5  → gyro_y     (Y축 자이로)
          6  → gyro_z     (Z축 자이로)
          7  → created_at (Unix초 × 1000)
          8  → angle_x    (X축 각도)
          9  → angle_y    (Y축 각도)
          10 → event_code (이벤트 코드)

        특수 로직: ID=10 이 없으면 event_code = 0
        """
        for tlv in tlv_items:
            if tlv.id == 1:
                data["acc_x_g"] = tlv.value
            elif tlv.id == 2:
                data["acc_y_g"] = tlv.value
            elif tlv.id == 3:
                data["acc_z_g"] = tlv.value
            elif tlv.id == 4:
                data["gyro_x_dps"] = tlv.value
            elif tlv.id == 5:
                data["gyro_y_dps"] = tlv.value
            elif tlv.id == 6:
                data["gyro_z_dps"] = tlv.value
            elif tlv.id == 7:
                if isinstance(tlv.value, int):
                    data["created_at"] = tlv.value * 1000
            elif tlv.id == 8:
                data["angle_x_deg"] = tlv.value
            elif tlv.id == 9:
                data["angle_y_deg"] = tlv.value
            elif tlv.id == 10:
                data["event_code"] = tlv.value

        # 특수 로직: ID=10 이 없으면 event_code 기본값 0
        # Go: hasEventCode := false; for _, tlv := range tlvItems { if tlv.ID == 10 { hasEventCode = true; break } }
        has_event_code = any(tlv.id == 10 for tlv in tlv_items)
        if not has_event_code:
            data["event_code"] = 0

        return data

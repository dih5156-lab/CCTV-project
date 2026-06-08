"""
tlv/transformer_v0.py
=====================
Go 원본: aiot-tlv-parser/pkg/tlv/transformer_v0.go

신버전(V0) TLV 트랜스포머 모듈입니다.
각 테이블 ID에 맞게 TLV 아이템 목록을 딕셔너리로 변환합니다.
"""

import struct
from typing import Any, Dict


class TransformerV0:
    """
    신버전 TLV 데이터 변환기
    Go: type TransformerV0 struct{}

    parse_XXX 메서드들이 Go의 각 parse34950() 등 메서드에 1:1 대응됩니다.
    """

    def transform(self, table_name: int, data: Dict[str, Any], tlv_items: list) -> Dict[str, Any]:
        """
        테이블 ID에 따라 적절한 파싱 메서드 호출
        Go: func (t *TransformerV0) Transform(tableName int, data map[string]interface{}, tlvItems []TLVItem) (map[string]interface{}, error)
        """
        handlers = {
            3:     self._parse3,
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
            raise ValueError(f"unsupported table name: {table_name}")
        return handler(data, tlv_items)

    def _parse3(self, data: Dict[str, Any], tlv_items: list) -> Dict[str, Any]:
        """
        Object 3 - 디바이스 정보 파싱
        Go: func (t *TransformerV0) parse3(...)

        ID → 필드 매핑:
          0  → manufacturer             (제조사)
          1  → model_number             (모델 번호)
          3  → firmware_version         (펌웨어 버전)
          4  → reboot                   (재부팅)
          5  → factory_reset            (공장 초기화)
          9  → battery_level            (배터리 잔량)
          11 → error_code               (에러 코드)
          12 → reset_error_code         (에러 코드 초기화)
          16 → supported_binding_and_modes (지원 바인딩 모드)
          18 → hardware_version         (하드웨어 버전)
          20 → battery_status           (배터리 상태)
        """
        for tlv in tlv_items:
            if tlv.id == 0:
                data["manufacturer"] = tlv.value
            elif tlv.id == 1:
                data["model_number"] = tlv.value
            elif tlv.id == 3:
                data["firmware_version"] = tlv.value
            elif tlv.id == 4:
                data["reboot"] = tlv.value
            elif tlv.id == 5:
                data["factory_reset"] = tlv.value
            elif tlv.id == 9:
                data["battery_level_pct"] = tlv.value
            elif tlv.id == 11:
                data["error_code"] = tlv.value
            elif tlv.id == 12:
                data["reset_error_code"] = tlv.value
            elif tlv.id == 16:
                data["supported_binding_and_modes"] = tlv.value
            elif tlv.id == 18:
                data["hardware_version"] = tlv.value
            elif tlv.id == 20:
                data["battery_status"] = tlv.value
        return data

    def _parse34950(self, data: Dict[str, Any], tlv_items: list) -> Dict[str, Any]:
        """
        Object 34950 - 하천 모니터링 파싱
        Go: func (t *TransformerV0) parse34950(...)

        ID → 필드 매핑:
          0     → water_level      (수위)
          1     → flow_velocity    (유속)
          2     → rain_fall        (강수량)
          26241 → reporting_period (보고 주기, ms → 초 변환)
        """
        for tlv in tlv_items:
            if tlv.id == 0:
                data["water_level_m"] = tlv.value
            elif tlv.id == 1:
                data["flow_velocity_mps"] = tlv.value
            elif tlv.id == 2:
                data["rain_fall_mm"] = tlv.value
            elif tlv.id == 26241:
                # Go: float64(val) / 1000.0  (ms → 초)
                if isinstance(tlv.value, int):
                    data["reporting_period_s"] = tlv.value / 1000.0
        return data

    def _parse34952(self, data: Dict[str, Any], tlv_items: list) -> Dict[str, Any]:
        """
        Object 34952 - 침수 감지 파싱
        Go: func (t *TransformerV0) parse34952(...)

        ID → 필드 매핑:
          0     → flood_level      (침수 수위)
          26241 → reporting_period (ms → 초)
        """
        for tlv in tlv_items:
            if tlv.id == 0:
                data["flood_level_m"] = tlv.value
            elif tlv.id == 26241:
                if isinstance(tlv.value, int):
                    data["reporting_period_s"] = tlv.value / 1000.0
        return data

    def _parse34954(self, data: Dict[str, Any], tlv_items: list) -> Dict[str, Any]:
        """
        Object 34954 - 온습도 파싱
        Go: func (t *TransformerV0) parse34954(...)

        ID → 필드 매핑:
          0     → temperature      (온도)
          1     → humidity         (습도)
          26241 → reporting_period (ms → 초)
        """
        for tlv in tlv_items:
            if tlv.id == 0:
                data["temperature_c"] = tlv.value
            elif tlv.id == 1:
                data["humidity_pct"] = tlv.value
            elif tlv.id == 26241:
                if isinstance(tlv.value, int):
                    data["reporting_period_s"] = tlv.value / 1000.0
        return data

    def _parse34955(self, data: Dict[str, Any], tlv_items: list) -> Dict[str, Any]:
        """
        Object 34955 - 경사계 파싱
        Go: func (t *TransformerV0) parse34955(...)

        ID → 필드 매핑:
          0     → angle_x                   (X축 각도)
          1     → angle_y                   (Y축 각도)
          2     → reporting_angle_threshold  (각도 임계값)
          3     → relative_angle_value_reset (상대 각도 초기화)
          26241 → reporting_period          (ms → 초)
        """
        for tlv in tlv_items:
            if tlv.id == 0:
                # uint32 비트 → IEEE 754 float32 해석 (Go: math.Float32frombits)
                data["angle_x_deg"] = struct.unpack(">f", struct.pack(">I", tlv.value))[0] if isinstance(tlv.value, int) else tlv.value
            elif tlv.id == 1:
                data["angle_y_deg"] = struct.unpack(">f", struct.pack(">I", tlv.value))[0] if isinstance(tlv.value, int) else tlv.value
            elif tlv.id == 2:
                data["reporting_angle_threshold_deg"] = tlv.value
            elif tlv.id == 3:
                data["relative_angle_value_reset"] = tlv.value
            elif tlv.id == 26241:
                if isinstance(tlv.value, int):
                    data["reporting_period_s"] = tlv.value / 1000.0
        return data

    def _parse34956(self, data: Dict[str, Any], tlv_items: list) -> Dict[str, Any]:
        """
        Object 34956 - 화재 경보 파싱
        Go: func (t *TransformerV0) parse34956(...)

        ID → 필드 매핑:
          0     → fire_alarm       (화재 감지)
          26241 → reporting_period (보고 주기)
        """
        for tlv in tlv_items:
            if tlv.id == 0:
                data["fire_alarm"] = tlv.value
            elif tlv.id == 26241:
                data["reporting_period_s"] = tlv.value
        return data

    def _parse34957(self, data: Dict[str, Any], tlv_items: list) -> Dict[str, Any]:
        """
        Object 34957 - 복합 요약1 파싱
        Go: func (t *TransformerV0) parse34957(...)

        ID → 필드 매핑:
          0 → temperature (온도)
          1 → angle_x    (X축 각도)
          2 → angle_y    (Y축 각도)
          3 → event_code (이벤트 코드)

        특수 로직: angle_x 와 angle_y 가 모두 존재하면 event_code = 1
        Go: if angleX ok && angleY ok { data["event_code"] = 1 }
        """
        for tlv in tlv_items:
            if tlv.id == 0:
                data["temperature_c"] = tlv.value
            elif tlv.id == 1:
                data["angle_x_deg"] = tlv.value
            elif tlv.id == 2:
                data["angle_y_deg"] = tlv.value
            elif tlv.id == 3:
                data["event_code"] = tlv.value

        # 특수 로직: angle_x_deg, angle_y_deg 모두 있으면 이벤트 코드 강제 설정
        if data.get("angle_x_deg") is not None and data.get("angle_y_deg") is not None:
            data["event_code"] = 1

        return data

    def _parse34958(self, data: Dict[str, Any], tlv_items: list) -> Dict[str, Any]:
        """
        Object 34958 - 복합 요약2 파싱
        Go: func (t *TransformerV0) parse34958(...)

        ID → 필드 매핑:
          0 → acc_x      (X축 가속도)
          1 → acc_y      (Y축 가속도)
          2 → acc_z      (Z축 가속도)
          3 → gyro_x     (X축 자이로)
          4 → gyro_y     (Y축 자이로)
          5 → gyro_z     (Z축 자이로)
          6 → angle_x    (X축 각도)
          7 → angle_y    (Y축 각도)
          8 → event_code (이벤트 코드)
        """
        for tlv in tlv_items:
            if tlv.id == 0:
                data["acc_x_g"] = tlv.value
            elif tlv.id == 1:
                data["acc_y_g"] = tlv.value
            elif tlv.id == 2:
                data["acc_z_g"] = tlv.value
            elif tlv.id == 3:
                data["gyro_x_dps"] = tlv.value
            elif tlv.id == 4:
                data["gyro_y_dps"] = tlv.value
            elif tlv.id == 5:
                data["gyro_z_dps"] = tlv.value
            elif tlv.id == 6:
                data["angle_x_deg"] = tlv.value
            elif tlv.id == 7:
                data["angle_y_deg"] = tlv.value
            elif tlv.id == 8:
                data["event_code"] = tlv.value
        return data

"""
tests/test_tlv_parser.py
========================
Go 원본: aiot-tlv-parser/tests/tlv_test.go

Go TestTLVParser 와 동일한 hex 입력 데이터를 사용해
Python 파서가 올바르게 동작하는지 검증합니다.

실행 방법:
    cd parser-python
    pytest tests/test_tlv_parser.py -v
"""

import sys
import os

# parser-python/ 폴더를 sys.path 에 추가 (절대 임포트 허용)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import binascii
import pytest

from tlv.parser import Parser, ParsedData


# ──────────────────────────────────────────────────────────────────
# 테스트 데이터  (Go: hexStrings 배열과 동일)
# ──────────────────────────────────────────────────────────────────
HEX_INPUTS = [
    # 1. Table 3   – 디바이스 정보
    "0033591000000300c300534157c2015341c403312e3032c10964c10b00c11041c2123031c11400",
    # 2. Table 34950 – 하천 모니터링 (수위/유속/강수량)
    "00068c1000888600c4003ec20c4ac40100000000c40200000000e4158e68c0c196",
    # 3. Table 34954 – 온도/습도
    "0065761000888a00c40041a80000c40142a00000e4668100000078e4158e68c0c24f",
    # 4. Table 34955 – 경사계
    "0004641000888b00c40042a6c358c4013f5bef21e4668100000e10c4020000000fe4158e68c0c18e",
    # 5. Table 34956 – 화재 경보
    "00b43c1000888c00c10000e4668100000078e4158e68c0c33f",
    # 6. Table 34957 – 복합 요약1 (온도+경사)
    "0008911000888d00c40041d85800c40142b0d627c4023fcd9413e4158e68c0c192",
    # 7. Table 34958 – 복합 요약2 (가속도+자이로+경사)
    "00035b1000888e00c4003bbfe3b0c4013f8150ecc4023ca467bec40300000000c40400000000c40500000000e4158e68c0c192",
]

# Go: hexStrings2 – 시각 비교용 원본 참조값 (assertions 의 기준)
EXPECTED_LABELS = [
    '{"error_code": 0, "manufacturer": "SAW", "model_number": "SA", "battery_level_pct": 100, "battery_status": 0, "firmware_version": "1.02", "supported_binding_and_modes": "A"}',
    '{"rain_fall_mm": 0, "created_at": 1757462934000, "water_level_m": 0.3790000081062317, "flow_velocity_mps": 0}',
    '{"humidity_pct": 80, "created_at": 1757463119000, "temperature_c": 21, "reporting_period_s": 120}',
    '{"angle_x_deg": 83.38153076171875, "angle_y_deg": 0.8591175675392151, "created_at": 1757462926000, "reporting_period_s": 3600, "reporting_angle_threshold_deg": 15}',
    '{"created_at": 1757463359000, "fire_alarm": false, "reporting_period": 120}',
    '{"angle_x_deg": 88.41826629638672, "angle_y_deg": 1.6060813665390015, "created_at": 1757462930000, "temperature_c": 27.04296875}',
    '{"acc_x_g": 0.0058559998869895935, "acc_y_g": 1.010282039642334, "acc_z_g": 0.020068999379873276, "gyro_x_dps": 0, "gyro_y_dps": 0, "gyro_z_dps": 0, "created_at": 1757462930000}',
]


@pytest.fixture(scope="module")
def parser():
    """Parser 인스턴스 – 모듈 내 모든 테스트에서 공유합니다."""
    return Parser()


def decode_hex(hex_str: str) -> bytes:
    """hex 문자열 → bytes  (Go: hex.DecodeString)"""
    return binascii.unhexlify(hex_str)


# ──────────────────────────────────────────────────────────────────
# 헬퍼: 파싱 결과를 출력 (Go: fmt.Printf 와 동일한 역할)
# ──────────────────────────────────────────────────────────────────
def print_result(index: int, result: ParsedData):
    print(f"\n=== TLV Parsing Result ===")
    print(f"Table Name : {result.table_name}")
    print(f"Data       : {result.data}")
    print(f"origin result: {EXPECTED_LABELS[index]}")


# ──────────────────────────────────────────────────────────────────
# Test Case 1  –  Table 3  (디바이스 정보)
# Go: Test Case 1
# ──────────────────────────────────────────────────────────────────
def test_case1_table3_device_info(parser):
    """
    Table 3 파싱 검증.
    expected: manufacturer=SAW, model_number=SA, firmware_version=1.02,
              battery_level=100, error_code=0/False,
              supported_binding_and_modes=A, battery_status=0/False
    """
    data = decode_hex(HEX_INPUTS[0])
    result = parser.decode_lwm2m_tlv(data, 8)
    print_result(0, result)

    assert result is not None
    assert result.table_name == "t3"

    d = result.data
    assert d.get("manufacturer") == "SAW"
    assert d.get("model_number") == "SA"
    assert d.get("firmware_version") == "1.02"
    assert d.get("battery_level_pct") == pytest.approx(100.0, rel=1e-3)
    # Go: error_code = 0 (int) / Python: False → False == 0 is True
    assert d.get("error_code") == 0
    assert d.get("supported_binding_and_modes") == "A"
    assert d.get("battery_status") == 0


# ──────────────────────────────────────────────────────────────────
# Test Case 2  –  Table 34950  (하천 모니터링)
# Go: Test Case 2
# ──────────────────────────────────────────────────────────────────
def test_case2_table34950_river(parser):
    """
    Table 34950 파싱 검증.
    expected: water_level≈0.379, flow_velocity=0, rain_fall=0
    """
    data = decode_hex(HEX_INPUTS[1])
    result = parser.decode_lwm2m_tlv(data, 8)
    print_result(1, result)

    assert result is not None
    assert result.table_name == "t34950"

    d = result.data
    assert d.get("water_level_m") == pytest.approx(0.3790000081062317, rel=1e-5)
    assert d.get("flow_velocity_mps") == pytest.approx(0.0, abs=1e-6)
    assert d.get("rain_fall_mm") == pytest.approx(0.0, abs=1e-6)


# ──────────────────────────────────────────────────────────────────
# Test Case 3  –  Table 34954  (온도/습도)
# Go: Test Case 3
# ──────────────────────────────────────────────────────────────────
def test_case3_table34954_temp_humidity(parser):
    """
    Table 34954 파싱 검증.
    expected: temperature≈21, humidity≈80
    """
    data = decode_hex(HEX_INPUTS[2])
    result = parser.decode_lwm2m_tlv(data, 8)
    print_result(2, result)

    assert result is not None
    assert result.table_name == "t34954"

    d = result.data
    assert d.get("temperature_c") == pytest.approx(21.0, rel=1e-3)
    assert d.get("humidity_pct") == pytest.approx(80.0, rel=1e-3)


# ──────────────────────────────────────────────────────────────────
# Test Case 4  –  Table 34955  (경사계)
# Go: Test Case 4
# ──────────────────────────────────────────────────────────────────
def test_case4_table34955_inclinometer(parser):
    """
    Table 34955 파싱 검증.
    expected: angle_x≈83.38, angle_y≈0.859, reporting_angle_threshold=15
    """
    data = decode_hex(HEX_INPUTS[3])
    result = parser.decode_lwm2m_tlv(data, 8)
    print_result(3, result)

    assert result is not None
    assert result.table_name == "t34955"

    d = result.data
    assert d.get("angle_x_deg") == pytest.approx(83.38153076171875, rel=1e-5)
    assert d.get("angle_y_deg") == pytest.approx(0.8591175675392151, rel=1e-5)
    assert d.get("reporting_angle_threshold_deg") == 15


# ──────────────────────────────────────────────────────────────────
# Test Case 5  –  Table 34956  (화재 경보)
# Go: Test Case 5
# ──────────────────────────────────────────────────────────────────
def test_case5_table34956_fire_alarm(parser):
    """
    Table 34956 파싱 검증.
    expected: fire_alarm=False
    """
    data = decode_hex(HEX_INPUTS[4])
    result = parser.decode_lwm2m_tlv(data, 8)
    print_result(4, result)

    assert result is not None
    assert result.table_name == "t34956"

    d = result.data
    # Go: fire_alarm = false / Python: False == 0
    assert d.get("fire_alarm") == False


# ──────────────────────────────────────────────────────────────────
# Test Case 6  –  Table 34957  (복합 요약1: 온도+경사)
# Go: Test Case 6
# ──────────────────────────────────────────────────────────────────
def test_case6_table34957_summary1(parser):
    """
    Table 34957 파싱 검증.
    expected: temperature≈27.04, angle_x≈88.42, angle_y≈1.606
    """
    data = decode_hex(HEX_INPUTS[5])
    result = parser.decode_lwm2m_tlv(data, 8)
    print_result(5, result)

    assert result is not None
    assert result.table_name == "t34957"

    d = result.data
    assert d.get("temperature_c") == pytest.approx(27.04296875, rel=1e-5)
    assert d.get("angle_x_deg") == pytest.approx(88.41826629638672, rel=1e-5)
    assert d.get("angle_y_deg") == pytest.approx(1.6060813665390015, rel=1e-5)


# ──────────────────────────────────────────────────────────────────
# Test Case 7  –  Table 34958  (복합 요약2: 가속도+자이로+경사)
# Go: Test Case 7
# ──────────────────────────────────────────────────────────────────
def test_case7_table34958_summary2(parser):
    """
    Table 34958 파싱 검증.
    expected: acc_x≈0.00586, acc_y≈1.010, acc_z≈0.0201,
              gyro_x=0, gyro_y=0, gyro_z=0
    """
    data = decode_hex(HEX_INPUTS[6])
    result = parser.decode_lwm2m_tlv(data, 8)
    print_result(6, result)

    assert result is not None
    assert result.table_name == "t34958"

    d = result.data
    assert d.get("acc_x_g") == pytest.approx(0.0058559998869895935, rel=1e-4)
    assert d.get("acc_y_g") == pytest.approx(1.010282039642334, rel=1e-5)
    assert d.get("acc_z_g") == pytest.approx(0.020068999379873276, rel=1e-4)
    assert d.get("gyro_x_dps") == pytest.approx(0.0, abs=1e-6)
    assert d.get("gyro_y_dps") == pytest.approx(0.0, abs=1e-6)
    assert d.get("gyro_z_dps") == pytest.approx(0.0, abs=1e-6)


# ──────────────────────────────────────────────────────────────────
# 파라미터화 테스트: 7개 케이스 모두 파싱 성공 여부만 확인
# (Go: for i, hexString := range hexStrings { testDecodeLwM2MTLV(...) })
# ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("hex_str, label", zip(HEX_INPUTS, EXPECTED_LABELS))
def test_all_parse_without_error(parser, hex_str, label):
    """7개 케이스 모두 예외 없이 파싱되는지 확인합니다."""
    data = decode_hex(hex_str)
    result = parser.decode_lwm2m_tlv(data, 8)
    assert result is not None, f"파싱 결과가 None입니다. label={label}"
    assert isinstance(result.table_name, str)
    assert isinstance(result.data, dict)


# ──────────────────────────────────────────────────────────────────
# 에러 케이스 테스트
# ──────────────────────────────────────────────────────────────────
def test_buffer_too_short(parser):
    """버퍼가 7바이트 미만이면 ValueError 가 발생해야 합니다."""
    with pytest.raises(ValueError, match="buffer too short"):
        parser.decode_lwm2m_tlv(bytes([0x00, 0x01, 0x02]), 8)


def test_disallowed_table(parser):
    """허용되지 않은 테이블 ID는 ValueError 가 발생해야 합니다."""
    buf = bytes([0x00] * 7)  # byte[5]=0, byte[6]=0 → table 0 (not allowed)
    with pytest.raises(ValueError, match="not allowed"):
        parser.decode_lwm2m_tlv(buf, 8)

"""
tlv/parser.py
=============
Go 원본: aiot-tlv-parser/pkg/tlv/parser.go

LwM2M TLV(Type-Length-Value) 바이너리 프로토콜 파서 모듈입니다.
Go의 encoding/binary + math → Python의 struct 모듈로 변환되었습니다.

TLV 구조 개요:
  [TypeByte(1) | ID(1 or 2) | Value(N bytes)]
  TypeByte 값에 따라 Value의 길이와 해석 방법이 달라집니다.

허용 테이블 ID:
  3, 34950, 34952, 34954, 34955, 34956, 34957, 34958
"""

import struct
import logging
import math
from typing import Optional, List, Any, Tuple

from tlv.transformer_v0 import TransformerV0
from tlv.transformer_v1 import TransformerV1

logger = logging.getLogger(__name__)

# 허용된 테이블 ID 목록
# Go: var AllowedTable = []int{3, 34950, 34952, 34954, 34955, 34956, 34957, 34958}
ALLOWED_TABLE = [3, 34950, 34952, 34954, 34955, 34956, 34957, 34958]


class TLVItem:
    """
    단일 TLV 아이템 (Type-Length-Value의 파싱 결과)
    Go: type TLVItem struct { ID int; Value interface{} }

    - id    : TLV 리소스 ID (정수)
    - value : 파싱된 값 (타입: bool, float, str, int, bytes 등)
    """
    def __init__(self, id: int, value: Any):
        self.id = id
        self.value = value

    def to_dict(self) -> dict:
        return {"id": self.id, "value": self.value}


class ParsedData:
    """
    TLV 파싱 결과 컨테이너
    Go: type ParsedData struct { TableName string; Data map[string]interface{} }

    - table_name : "t34950" 형태의 테이블 이름 문자열
    - data       : 파싱된 필드명:값 딕셔너리
    """
    def __init__(self, table_name: str, data: dict):
        self.table_name = table_name
        self.data = data


class Parser:
    """
    LwM2M TLV 파서 메인 클래스
    Go: type Parser struct { transformerV0 *TransformerV0; transformerV1 *TransformerV1 }
    """

    def __init__(self):
        """
        Go: func NewParser() *Parser
        트랜스포머 두 버전을 모두 생성하여 보유합니다.
        """
        self._transformer_v0 = TransformerV0()
        self._transformer_v1 = TransformerV1()

    def decode_lwm2m_tlv(self, buffer: bytes, start_index: int) -> Optional[ParsedData]:
        """
        LwM2M TLV 바이너리 데이터를 파싱합니다.
        Go: func (p *Parser) DecodeLwM2MTLV(buffer []byte, startIndex int) (*ParsedData, error)

        Args:
            buffer      : 수신된 바이너리 페이로드 전체
            start_index : TLV 데이터 시작 오프셋 (보통 8)

        Returns:
            ParsedData 또는 None (파싱 실패 / 허용되지 않은 테이블)
        """
        if len(buffer) < 7:
            raise ValueError(f"buffer too short: expected at least 7 bytes, got {len(buffer)}")

        # 바이트 5,6에서 테이블 ID 추출 (Big-Endian uint16)
        # Go: decimalValue := int(binary.BigEndian.Uint16([]byte{buffer[5], buffer[6]}))
        decimal_value = struct.unpack(">H", bytes([buffer[5], buffer[6]]))[0]

        # 테이블 허용 여부 확인
        if not _is_table_allowed(decimal_value):
            raise ValueError(f"table {decimal_value} not allowed")

        # 구버전 여부: 첫 바이트가 ASCII '1' (0x31) 이면 구버전
        # Go: oldVersion := len(buffer) > 0 && buffer[0] == '1'
        old_version = len(buffer) > 0 and buffer[0] == ord('1')

        return self._decode_tlv(decimal_value, buffer[start_index:], old_version)

    def _decode_tlv(self, table_name: int, buffer: bytes, old_version: bool) -> Optional[ParsedData]:
        """
        테이블별 TLV 디코딩 내부 로직
        Go: func (p *Parser) decodeTLV(tableName int, buffer []byte, oldVersion bool) (*ParsedData, error)
        """
        # 구버전에서 테이블 3은 지원하지 않음
        if old_version and table_name == 3:
            raise ValueError("table 3 not supported for old version")

        # 구버전 34952 → 34954로 테이블 변환
        # Go: if tableName == 34952 && oldVersion { tableName = 34954 }
        if table_name == 34952 and old_version:
            table_name = 34954

        # TLV 아이템 파싱
        tlv_items = self._parse_tlv_items(table_name, buffer)

        # 기본 데이터 구조
        data = {"tableName": f"t{table_name}"}

        # 버전에 따른 트랜스포머 선택
        if old_version:
            transformed_data = self._transformer_v1.transform(table_name, data, tlv_items)
        else:
            transformed_data = self._transformer_v0.transform(table_name, data, tlv_items)

        return ParsedData(
            table_name=transformed_data["tableName"],
            data=transformed_data,
        )

    def _parse_tlv_items(self, table_name: int, buffer: bytes) -> List[TLVItem]:
        """
        버퍼에서 TLV 아이템 목록을 순차적으로 파싱
        Go: func (p *Parser) parseTLVItems(tableName int, buffer []byte) ([]TLVItem, error)
        """
        items: List[TLVItem] = []
        index = 0

        while index < len(buffer):
            if index >= len(buffer):
                break

            type_byte = buffer[index]
            index += 1

            # ID 읽기
            # Go: if typeByte != 0xe4 { id = int(buffer[index]) } else { id = BigEndian.Uint16(...) }
            if type_byte != 0xe4:
                if index >= len(buffer):
                    raise ValueError("buffer overflow while reading ID")
                id_ = buffer[index]
                index += 1
            else:
                if index + 1 >= len(buffer):
                    raise ValueError("buffer overflow while reading 16-bit ID")
                id_ = struct.unpack(">H", buffer[index:index + 2])[0]
                index += 2

            # 값 읽기
            try:
                value, value_length = self._read_value(table_name, id_, type_byte, buffer, index)
            except Exception as e:
                logger.warning(f"Warning: failed to read value for type {type_byte:02x}, id {id_}: {e}")
                continue

            index += value_length
            items.append(TLVItem(id=id_, value=value))

        return items

    def _read_value(self, table_name: int, id_: int, type_byte: int, buffer: bytes, index: int) -> Tuple[Any, int]:
        """
        타입 바이트에 따라 버퍼에서 값을 읽음
        Go: func (p *Parser) readValue(tableName int, id int, typeByte byte, buffer []byte, index int) (interface{}, int, error)

        타입 바이트 의미:
          0xc1 : 1바이트 값 (bool 또는 특수 정수)
          0xc2 : 2바이트 문자열
          0xc3 : 3바이트 문자열
          0xc4 : 4바이트 float32 (또는 uint32)
          0xc5 : 5바이트 문자열
          0xe4 : 4바이트 Unix timestamp (uint32)

        Returns:
            (value, value_length) 튜플
        """
        if type_byte == 0xc1:  # 1바이트 값
            if index >= len(buffer):
                raise ValueError("buffer overflow for 1-byte value")
            # 테이블 3의 특수 처리
            if table_name == 3:
                if id_ == 9:
                    return float(buffer[index]), 1
                if id_ == 16:
                    return chr(buffer[index]), 1
            # 일반: 0이 아니면 True
            return buffer[index] != 0, 1

        elif type_byte == 0xc2:  # 2바이트 문자열
            if index + 1 >= len(buffer):
                raise ValueError("buffer overflow for 2-byte value")
            return buffer[index:index + 2].decode("latin-1", errors="replace"), 2

        elif type_byte == 0xc3:  # 3바이트 문자열
            if index + 2 >= len(buffer):
                raise ValueError("buffer overflow for 3-byte value")
            return buffer[index:index + 3].decode("latin-1", errors="replace"), 3

        elif type_byte == 0xc4:  # 4바이트 float32 또는 uint32
            if index + 3 >= len(buffer):
                raise ValueError("buffer overflow for 4-byte value")

            bits = struct.unpack(">I", buffer[index:index + 4])[0]  # Big-Endian uint32

            # 테이블 3, ID=3: 문자열로 처리
            if table_name == 3 and id_ == 3:
                return buffer[index:index + 4].decode("latin-1", errors="replace"), 4

            # 테이블 34955: uint32 그대로 반환
            if table_name == 34955:
                return bits, 4

            # 일반: IEEE 754 float32로 해석
            # Go: math.Float32frombits(bits)
            value = struct.unpack(">f", buffer[index:index + 4])[0]
            return float(value), 4

        elif type_byte == 0xc5:  # 5바이트 문자열
            if index + 4 >= len(buffer):
                raise ValueError("buffer overflow for 5-byte value")
            return buffer[index:index + 5].decode("latin-1", errors="replace"), 5

        elif type_byte == 0xe4:  # 4바이트 Unix timestamp
            if index + 3 >= len(buffer):
                raise ValueError("buffer overflow for timestamp")
            # Go: int64(binary.BigEndian.Uint32(buffer[index:index+4]))
            timestamp = struct.unpack(">I", buffer[index:index + 4])[0]
            return int(timestamp), 4

        else:
            raise ValueError(f"unknown type byte: {type_byte:02x}")


def _is_table_allowed(table_id: int) -> bool:
    """
    테이블 ID가 허용 목록에 있는지 확인
    Go: func isTableAllowed(tableID int) bool
    """
    return table_id in ALLOWED_TABLE

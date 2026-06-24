"""H264 POC 보정 유틸리티.

DeepStream H264 출력 경로에서 MediaMTX DTS 계산 문제를 피하기 위해
슬라이스 헤더의 poc_lsb 값을 순차값으로 보정한다.
"""

from __future__ import annotations

from threading import Lock


class H264PocFixer:
    """H264 Annex B 스트림의 poc_lsb 값을 순차적으로 수정한다."""

    def __init__(self) -> None:
        self._log2_max_frame_num: int = 8
        self._poc_lsb_bits: int = 8
        self._poc_counter: int = 0
        self._lock = Lock()

    @staticmethod
    def _get_bit(data: bytes, pos: int) -> int:
        return (data[pos >> 3] >> (7 - (pos & 7))) & 1

    @staticmethod
    def _set_bit(data: bytearray, pos: int, bit: int) -> None:
        idx = pos >> 3
        shift = 7 - (pos & 7)
        if bit:
            data[idx] |= 1 << shift
        else:
            data[idx] &= ~(1 << shift)

    @classmethod
    def _read_ue(cls, data: bytes, pos: list) -> int:
        """Exp-Golomb ue(v) 읽기."""
        m = 0
        while pos[0] < len(data) * 8 and not cls._get_bit(data, pos[0]):
            m += 1
            pos[0] += 1
        pos[0] += 1
        val = (1 << m) - 1
        for i in range(m - 1, -1, -1):
            val += cls._get_bit(data, pos[0]) << i
            pos[0] += 1
        return val

    @classmethod
    def _read_u(cls, data: bytes, pos: list, n: int) -> int:
        """고정 n 비트 부호 없는 정수 읽기."""
        val = 0
        for _ in range(n):
            val = (val << 1) | cls._get_bit(data, pos[0])
            pos[0] += 1
        return val

    @classmethod
    def _write_u(cls, data: bytearray, bit_pos: int, n: int, val: int) -> None:
        """고정 n 비트 부호 없는 정수 쓰기."""
        for i in range(n - 1, -1, -1):
            cls._set_bit(data, bit_pos, (val >> i) & 1)
            bit_pos += 1

    def _parse_sps(self, nalu: bytes) -> None:
        """SPS NAL 유닛에서 log2_max_frame_num 및 poc_lsb_bits 추출."""
        try:
            body = nalu[4:]
            pos = [0]
            self._read_ue(body, pos)
            self._log2_max_frame_num = self._read_ue(body, pos) + 4
            poc_type = self._read_ue(body, pos)

            if poc_type == 0:
                self._poc_lsb_bits = self._read_ue(body, pos) + 4
            else:
                self._poc_lsb_bits = 4
        except Exception:
            pass

    def _poc_lsb_bit_pos(self, nalu: bytes, is_idr: bool):
        """슬라이스 NALU RBSP에서 poc_lsb 필드 시작 비트 위치 반환."""
        try:
            body = nalu[1:]
            pos = [0]
            self._read_ue(body, pos)
            self._read_ue(body, pos)
            self._read_ue(body, pos)
            self._read_u(body, pos, self._log2_max_frame_num)
            if is_idr:
                self._read_ue(body, pos)
            return pos[0], body
        except Exception:
            return None, None

    @staticmethod
    def _iter_nals(data: bytes):
        """Annex B 바이트 스트림에서 (start_offset, end_offset) 쌍을 생성."""
        starts = []
        i = 0
        n = len(data)
        while i < n - 2:
            if i + 3 < n and data[i : i + 4] == b"\x00\x00\x00\x01":
                starts.append(i + 4)
                i += 4
            elif data[i : i + 3] == b"\x00\x00\x01":
                starts.append(i + 3)
                i += 3
            else:
                i += 1
        for j, s in enumerate(starts):
            if j + 1 < len(starts):
                e = starts[j + 1]
                e -= 4 if (e >= 4 and data[e - 4 : e] == b"\x00\x00\x00\x01") else 3
            else:
                e = n
            yield s, e

    def process_buffer(self, data: bytearray) -> None:
        """H264 버퍼 전체를 스캔하여 슬라이스 헤더의 poc_lsb를 순차값으로 수정."""
        raw = bytes(data)
        for s, e in self._iter_nals(raw):
            if s >= e or s >= len(raw):
                continue
            nal_type = raw[s] & 0x1F
            nalu = raw[s:e]

            if nal_type == 7:
                self._parse_sps(nalu)
            elif nal_type == 5:
                with self._lock:
                    self._poc_counter = 2
            elif nal_type == 1:
                with self._lock:
                    target_poc = self._poc_counter
                    self._poc_counter = (self._poc_counter + 2) % (
                        1 << self._poc_lsb_bits
                    )

                bit_pos, _body = self._poc_lsb_bit_pos(nalu, False)
                if bit_pos is not None:
                    data_bit_pos = (s + 1) * 8 + bit_pos
                    self._write_u(data, data_bit_pos, self._poc_lsb_bits, target_poc)

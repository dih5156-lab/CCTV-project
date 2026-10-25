import random

import pytest

from src.core._h264_poc_fixer import H264PocFixer


@pytest.mark.parametrize("data,expected", [
    (b"", []),
    (b"\x00\x00", []),
    (b"\x00\x00\x02\x01", []),
    (b"\x00\x00\x01", [(3, 3)]),
    (b"\x00\x00\x00\x01", [(4, 4)]),
    (b"\x00" * 5 + b"\x01\x09", [(6, 7)]),
    (b"\x00\x00\x01\x00\x00\x00\x01", [(3, 3), (7, 7)]),
    (b"\x00\x00\x01\x09\xf0\x00\x00\x00\x01\x65\x88\x00\x00\x01\x41",
     [(3, 5), (9, 11), (14, 15)]),
    (b"\x00\x00\x01\x41\x00\x00", [(3, 6)]),
])
def test_annex_b_boundaries_preserve_empty_mixed_and_trailing_units(data, expected):
    assert list(H264PocFixer._iter_nals(data)) == expected


def test_annex_b_search_matches_previous_scan_for_overlapping_zero_sequences():
    # 구형 순회를 독립 기준으로 남겨 시작 코드가 겹치는 입력의 경계 변경을 잡는다.
    def previous_scan(data):
        starts = []
        index = 0
        while index < len(data) - 2:
            if data[index:index + 4] == b"\x00\x00\x00\x01":
                starts.append((index, index + 4))
                index += 4
            elif data[index:index + 3] == b"\x00\x00\x01":
                starts.append((index, index + 3))
                index += 3
            else:
                index += 1
        return [(start, starts[n + 1][0] if n + 1 < len(starts) else len(data))
                for n, (_, start) in enumerate(starts)]

    rng = random.Random(20261019)
    for _ in range(1000):
        data = bytes(rng.choices([0, 0, 0, 1, 2, 3, 65, 101, 255], k=rng.randrange(512)))
        assert list(H264PocFixer._iter_nals(data)) == previous_scan(data)


def test_poc_fix_preserves_prefixes_and_resets_at_idr_across_buffers():
    fixer = H264PocFixer()
    idr = b"\x00\x00\x00\x01\x65\xe0\x00\x00"
    slice_prefix = b"\x00\x00\x01\x41\xe0\x00"
    first = bytearray(idr + slice_prefix + b"\x00")
    fixer.process_buffer(first)
    assert first == idr + slice_prefix + b"\x40"
    second = bytearray(slice_prefix + b"\x00")
    fixer.process_buffer(second)
    assert second == slice_prefix + b"\x80"
    reset = bytearray(idr + slice_prefix + b"\x00")
    fixer.process_buffer(reset)
    assert reset == first

"""DeepStream tensor meta utility tests."""

from __future__ import annotations

import ctypes
from types import SimpleNamespace

import pytest

from src.core._deepstream_tensor_utils import read_pphuman_obj_scores


class _Cast:
    @staticmethod
    def cast(value):
        return value


class _Pyds:
    NVDSINFER_TENSOR_OUTPUT_META = 100

    class NvDsInferDataType:
        FLOAT = 0
        HALF = 1

    NvDsUserMeta = _Cast
    NvDsInferTensorMeta = _Cast

    def __init__(self, layer):
        self._layer = layer

    def get_nvds_LayerInfo(self, tensor_meta, layer_idx):
        return self._layer

    @staticmethod
    def get_ptr(buffer):
        return ctypes.addressof(buffer)


def test_read_pphuman_obj_scores_falls_back_to_26_score_tensor_without_gie_id():
    values = (ctypes.c_float * 26)(*[0.1] * 26)
    layer = SimpleNamespace(
        inferDims=SimpleNamespace(numDims=1, d=[26]),
        dataType=0,
        buffer=values,
        layerName="scores",
    )
    tensor_meta = SimpleNamespace(num_output_layers=1)
    user_meta = SimpleNamespace(
        base_meta=SimpleNamespace(meta_type=100),
        user_meta_data=tensor_meta,
    )
    obj_meta = SimpleNamespace(obj_user_meta_list=SimpleNamespace(data=user_meta, next=None))

    scores = read_pphuman_obj_scores(
        obj_meta,
        pyds_module=_Pyds(layer),
        pphuman_gie_id=3,
        default_gie_id=1,
    )

    assert len(scores) == 26
    assert scores[0] == pytest.approx(0.1)

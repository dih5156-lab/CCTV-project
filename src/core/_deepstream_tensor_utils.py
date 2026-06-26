"""DeepStream tensor meta 처리 유틸리티."""

from __future__ import annotations

import ctypes
from typing import Any, List

import numpy as np


def layer_dims(layer: Any) -> List[int]:
    """NvDsInferLayerInfo에서 양수 dimension 목록을 읽는다."""
    dims = getattr(layer, "inferDims", None)
    if dims is None:
        return []
    num_dims = int(getattr(dims, "numDims", 0) or 0)
    values = []
    for idx in range(num_dims):
        value = int(dims.d[idx])
        if value > 0:
            values.append(value)
    return values


def layer_to_numpy(layer: Any, pyds_module: Any) -> Any:
    """NvDsInferLayerInfo buffer를 float32 numpy 배열로 변환한다."""
    dims = layer_dims(layer)
    if not dims:
        return None
    size = 1
    for dim in dims:
        size *= dim
    if size <= 0:
        return None

    data_type = int(getattr(layer, "dataType", 0))
    if data_type == int(pyds_module.NvDsInferDataType.FLOAT):
        c_type = ctypes.c_float
        dtype = np.float32
    elif data_type == int(pyds_module.NvDsInferDataType.HALF):
        c_type = ctypes.c_uint16
        dtype = np.float16
    else:
        return None

    ptr = pyds_module.get_ptr(layer.buffer)
    array_type = c_type * size
    raw = array_type.from_address(ptr)
    return np.ctypeslib.as_array(raw).view(dtype).reshape(dims).astype(
        np.float32,
        copy=False,
    )


def select_yolo_output(tensor_meta: Any, pyds_module: Any) -> Any:
    """YOLO 출력으로 보이는 tensor layer를 선택한다."""
    for layer_idx in range(int(tensor_meta.num_output_layers)):
        layer = pyds_module.get_nvds_LayerInfo(tensor_meta, layer_idx)
        dims = layer_dims(layer)
        if len(dims) >= 2 and 4 < min(dims[-2:]) and max(dims[-2:]) >= 100:
            name = getattr(layer, "layerName", "") or ""
            if not name or "output" in str(name):
                return layer
    return None


def select_pphuman_layer(tensor_meta: Any, pyds_module: Any) -> Any:
    """PP-Human SGIE tensor에서 fetch_name_0 레이어를 선택한다."""
    for layer_idx in range(int(tensor_meta.num_output_layers)):
        layer = pyds_module.get_nvds_LayerInfo(tensor_meta, layer_idx)
        name = str(getattr(layer, "layerName", "") or "")
        if "fetch_name_0" in name:
            return layer
    if int(tensor_meta.num_output_layers) > 0:
        return pyds_module.get_nvds_LayerInfo(tensor_meta, 0)
    return None


def tensor_gie_id(tensor_meta: Any, default_gie_id: int) -> int:
    """Tensor meta의 GIE id를 읽고 없으면 기본 id를 반환한다."""
    for attr_name in ("unique_id", "gie_unique_id"):
        value = getattr(tensor_meta, attr_name, None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    return default_gie_id


def read_pphuman_obj_scores(
    obj_meta: Any,
    *,
    pyds_module: Any,
    pphuman_gie_id: int,
    default_gie_id: int,
) -> List[float]:
    """NvDsObjectMeta.obj_user_meta_list에서 PP-Human score 목록을 추출한다."""
    fallback_scores: List[float] = []
    l_user = obj_meta.obj_user_meta_list
    while l_user is not None:
        try:
            user_meta = pyds_module.NvDsUserMeta.cast(l_user.data)
        except StopIteration:
            break
        if user_meta.base_meta.meta_type == pyds_module.NVDSINFER_TENSOR_OUTPUT_META:
            tensor_meta = pyds_module.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
            layer = select_pphuman_layer(tensor_meta, pyds_module)
            if layer is not None:
                output = layer_to_numpy(layer, pyds_module)
                if output is not None:
                    scores = output.reshape(-1).tolist()
                    if tensor_gie_id(tensor_meta, default_gie_id) == pphuman_gie_id:
                        return scores
                    # 일부 pyds 버전에서는 SGIE object tensor의 gie id가 노출되지 않는다.
                    # PA100K/PP-Human 속성 출력은 26-score 벡터라 안전한 fallback 후보로 둔다.
                    if len(scores) == 26 and not fallback_scores:
                        fallback_scores = scores
        try:
            l_user = l_user.next
        except StopIteration:
            break
    return fallback_scores

"""
geometry.py - 기하학 유틸리티 함수
"""

from typing import Tuple, Union
from ..core.events import DetectionEvent


def calculate_iou(box1: DetectionEvent, box2: DetectionEvent) -> float:
    """두 박스 간의 IoU (Intersection over Union) 계산"""
    if not isinstance(box1, DetectionEvent) or not isinstance(box2, DetectionEvent):
        raise TypeError("두 인자 모두 DetectionEvent 객체여야 함")
    
    bbox1 = {'x': box1.x, 'y': box1.y, 'width': box1.width, 'height': box1.height}
    bbox2 = {'x': box2.x, 'y': box2.y, 'width': box2.width, 'height': box2.height}
    return calculate_bbox_iou(bbox1, bbox2)


def boxes_overlap(box1: DetectionEvent, box2: DetectionEvent, threshold: float = 0.3) -> bool:
    """두 박스가 겹치는지 확인"""
    if not isinstance(box1, DetectionEvent) or not isinstance(box2, DetectionEvent):
        raise TypeError("두 인자 모두 DetectionEvent 객체여야 함")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"임계값은 0.0과 1.0 사이여야 함, 입력값: {threshold}")
    return calculate_iou(box1, box2) > threshold


# ===== bbox_utils.py에서 이전된 함수들 =====

def get_center(bbox: dict) -> Tuple[float, float]:
    """바운딩 박스의 중심점 계산"""
    if not isinstance(bbox, dict):
        raise TypeError("bbox는 딕셔너리여야 함")
    required_keys = ['x', 'y', 'width', 'height']
    if not all(k in bbox for k in required_keys):
        raise ValueError(f"bbox는 다음 키를 포함해야 함: {required_keys}")
    
    cx = bbox['x'] + bbox['width'] / 2
    cy = bbox['y'] + bbox['height'] / 2
    return cx, cy


def point_in_bbox(px: Union[int, float], py: Union[int, float], bbox: dict) -> bool:
    """점이 바운딩 박스 내부에 있는지 확인"""
    if not isinstance(bbox, dict):
        raise TypeError("bbox는 딕셔너리여야 함")
    required_keys = ['x', 'y', 'width', 'height']
    if not all(k in bbox for k in required_keys):
        raise ValueError(f"bbox는 다음 키를 포함해야 함: {required_keys}")
    
    return (
        bbox['x'] <= px <= bbox['x'] + bbox['width'] and
        bbox['y'] <= py <= bbox['y'] + bbox['height']
    )


def _calculate_intersection_area(bbox1: dict, bbox2: dict) -> float:
    """두 바운딩 박스의 교집합 면적 계산"""
    x1_min, y1_min = bbox1['x'], bbox1['y']
    x1_max = x1_min + bbox1['width']
    y1_max = y1_min + bbox1['height']
    
    x2_min, y2_min = bbox2['x'], bbox2['y']
    x2_max = x2_min + bbox2['width']
    y2_max = y2_min + bbox2['height']
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    return (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)


def calculate_bbox_iou(bbox1: dict, bbox2: dict) -> float:
    """두 바운딩 박스의 IoU (Intersection over Union) 계산"""
    if not isinstance(bbox1, dict) or not isinstance(bbox2, dict):
        raise TypeError("두 인자 모두 딕셔너리여야 함")
    
    inter_area = _calculate_intersection_area(bbox1, bbox2)
    
    if inter_area == 0.0:
        return 0.0
    
    area1 = bbox1['width'] * bbox1['height']
    area2 = bbox2['width'] * bbox2['height']
    union_area = area1 + area2 - inter_area
    
    if union_area <= 0:
        return 0.0
    
    return inter_area / union_area


def calculate_overlap_ratio(bbox1: dict, bbox2: dict) -> float:
    """bbox1 대비 bbox2의 겹침 비율 계산"""
    if not isinstance(bbox1, dict) or not isinstance(bbox2, dict):
        raise TypeError("두 인자 모두 딕셔너리여야 함")
    
    inter_area = _calculate_intersection_area(bbox1, bbox2)
    
    if inter_area == 0.0:
        return 0.0
    
    bbox2_area = bbox2['width'] * bbox2['height']
    
    if bbox2_area <= 0:
        return 0.0
    
    return inter_area / bbox2_area


def get_head_bbox(person_bbox: dict, head_ratio: float = 0.7) -> dict:
    """사람 바운딩 박스에서 머리 영역 추출"""
    if not isinstance(person_bbox, dict):
        raise TypeError("person_bbox는 딕셔너리여야 함")
    if not 0.0 < head_ratio <= 1.0:
        raise ValueError(f"head_ratio는 0.0과 1.0 사이여야 함, 입력값: {head_ratio}")
    
    return {
        'x': person_bbox['x'],
        'y': person_bbox['y'],
        'width': person_bbox['width'],
        'height': person_bbox['height'] * head_ratio
    }


def is_helmet_worn(person_bbox: dict, helmet_bboxes: list, 
                   head_ratio: float = 0.7, 
                   iou_threshold: float = 0.1,
                   overlap_threshold: float = 0.3) -> bool:
    """헬멧 착용 여부 판단"""
    if not isinstance(person_bbox, dict):
        raise TypeError("person_bbox는 딕셔너리여야 함")
    if not isinstance(helmet_bboxes, list):
        raise TypeError("helmet_bboxes는 리스트여야 함")
    if not 0.0 < head_ratio <= 1.0:
        raise ValueError(f"head_ratio는 0.0과 1.0 사이여야 함, 입력값: {head_ratio}")
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError(f"iou_threshold는 0.0과 1.0 사이여야 함, 입력값: {iou_threshold}")
    if not 0.0 <= overlap_threshold <= 1.0:
        raise ValueError(f"overlap_threshold는 0.0과 1.0 사이여야 함, 입력값: {overlap_threshold}")
    
    head_bbox = get_head_bbox(person_bbox, head_ratio)
    
    for helmet_bbox in helmet_bboxes:
        if not isinstance(helmet_bbox, dict):
            continue  # 잘못된 헬멧 박스는 건너띠
        
        try:
            cx, cy = get_center(helmet_bbox)
            if point_in_bbox(cx, cy, head_bbox):
                return True
            
            iou = calculate_bbox_iou(head_bbox, helmet_bbox)
            if iou >= iou_threshold:
                return True
            
            overlap_ratio = calculate_overlap_ratio(head_bbox, helmet_bbox)
            if overlap_ratio >= overlap_threshold:
                return True
        except (KeyError, ValueError, TypeError):
            continue  # 잘못된 헬멧 박스는 건너띠
    
    return False

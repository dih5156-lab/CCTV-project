"""
utils/geometry.py - 기하학 관련 유틸리티 함수
"""

from typing import Tuple
from ..core.events import DetectionEvent


def calculate_iou(box1: DetectionEvent, box2: DetectionEvent) -> float:
    """
    두 박스 간 IoU (Intersection over Union) 계산 (DetectionEvent 타입)
    
    Args:
        box1: 첫 번째 박스
        box2: 두 번째 박스
        
    Returns:
        IoU 값 (0.0 ~ 1.0)
    """
    # DetectionEvent를 dict로 변환하여 calculate_bbox_iou 호출
    bbox1 = {'x': box1.x, 'y': box1.y, 'width': box1.width, 'height': box1.height}
    bbox2 = {'x': box2.x, 'y': box2.y, 'width': box2.width, 'height': box2.height}
    return calculate_bbox_iou(bbox1, bbox2)


def calculate_box_area(box: DetectionEvent) -> int:
    """박스 면적 계산"""
    return box.width * box.height


def boxes_overlap(box1: DetectionEvent, box2: DetectionEvent, threshold: float = 0.3) -> bool:
    """
    두 박스가 겹치는지 확인
    
    Args:
        box1: 첫 번째 박스
        box2: 두 번째 박스
        threshold: IoU 임계값 (기본 0.3)
        
    Returns:
        겹침 여부
    """
    return calculate_iou(box1, box2) > threshold


# ===== bbox_utils.py에서 이전된 함수들 =====

def get_center(bbox: dict):
    """바운딩 박스의 중심점 계산 (dict 타입)"""
    cx = bbox['x'] + bbox['width'] / 2
    cy = bbox['y'] + bbox['height'] / 2
    return cx, cy


def point_in_bbox(px, py, bbox: dict) -> bool:
    """점이 바운딩 박스 안에 있는지 확인"""
    return (
        bbox['x'] <= px <= bbox['x'] + bbox['width'] and
        bbox['y'] <= py <= bbox['y'] + bbox['height']
    )


def _calculate_intersection_area(bbox1: dict, bbox2: dict) -> float:
    """두 바운딩 박스의 교집합 면적 계산 (내부 헬퍼 함수)"""
    x1_min, y1_min = bbox1['x'], bbox1['y']
    x1_max = x1_min + bbox1['width']
    y1_max = y1_min + bbox1['height']
    
    x2_min, y2_min = bbox2['x'], bbox2['y']
    x2_max = x2_min + bbox2['width']
    y2_max = y2_min + bbox2['height']
    
    # 교집합 영역
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    return (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)


def calculate_bbox_iou(bbox1: dict, bbox2: dict) -> float:
    """
    두 바운딩 박스의 IoU(Intersection over Union) 계산 (dict 타입)
    
    Args:
        bbox1: {'x', 'y', 'width', 'height'} 형식의 첫 번째 박스
        bbox2: {'x', 'y', 'width', 'height'} 형식의 두 번째 박스
        
    Returns:
        IoU 값 (0.0 ~ 1.0)
    """
    inter_area = _calculate_intersection_area(bbox1, bbox2)
    
    if inter_area == 0.0:
        return 0.0
    
    # 각 박스 면적
    area1 = bbox1['width'] * bbox1['height']
    area2 = bbox2['width'] * bbox2['height']
    
    # 합집합 영역
    union_area = area1 + area2 - inter_area
    
    if union_area <= 0:
        return 0.0
    
    return inter_area / union_area


def calculate_overlap_ratio(bbox1: dict, bbox2: dict) -> float:
    """bbox2가 bbox1과 겹치는 비율 계산 (bbox2 기준)"""
    inter_area = _calculate_intersection_area(bbox1, bbox2)
    
    if inter_area == 0.0:
        return 0.0
    
    bbox2_area = bbox2['width'] * bbox2['height']
    
    if bbox2_area <= 0:
        return 0.0
    
    return inter_area / bbox2_area


def get_head_bbox(person_bbox: dict, head_ratio: float = 0.7):
    """
    사람 바운딩 박스에서 머리 영역 추출
    head_ratio: 사람 박스 상단에서 차지하는 비율 (0.7 = 상단 70%)
    """
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
    """
    헬멧 착용 여부 판단 (개선된 버전)
    
    Args:
        person_bbox: 사람 바운딩 박스
        helmet_bboxes: 헬멧 바운딩 박스 리스트
        head_ratio: 머리 영역 비율 (기본 70% - 가까이서도 감지)
        iou_threshold: IoU 최소 임계값
        overlap_threshold: 헬멧이 머리 영역과 겹치는 최소 비율
    
    Returns:
        헬멧 착용 시 True
    """
    head_bbox = get_head_bbox(person_bbox, head_ratio)
    
    for helmet_bbox in helmet_bboxes:
        # 방법 1: 헬멧 중심점이 머리 영역 안에 있는 경우
        cx, cy = get_center(helmet_bbox)
        if point_in_bbox(cx, cy, head_bbox):
            return True
        
        # 방법 2: IoU가 임계값 이상인 경우
        iou = calculate_bbox_iou(head_bbox, helmet_bbox)
        if iou >= iou_threshold:
            return True
        
        # 방법 3: 헬멧이 머리 영역과 일정 비율 이상 겹치는 경우
        overlap_ratio = calculate_overlap_ratio(head_bbox, helmet_bbox)
        if overlap_ratio >= overlap_threshold:
            return True
    
    return False

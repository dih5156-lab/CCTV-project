"""
utils/geometry.py - 기하학 관련 유틸리티 함수
"""

from typing import Tuple
from events import DetectionEvent


def calculate_iou(box1: DetectionEvent, box2: DetectionEvent) -> float:
    """
    두 박스 간 IoU (Intersection over Union) 계산
    
    Args:
        box1: 첫 번째 박스
        box2: 두 번째 박스
        
    Returns:
        IoU 값 (0.0 ~ 1.0)
    """
    x1_min, y1_min = box1.x, box1.y
    x1_max, y1_max = box1.x + box1.width, box1.y + box1.height
    x2_min, y2_min = box2.x, box2.y
    x2_max, y2_max = box2.x + box2.width, box2.y + box2.height
    
    # 교집합 계산
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # 합집합 계산
    box1_area = box1.width * box1.height
    box2_area = box2.width * box2.height
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def calculate_box_center(box: DetectionEvent) -> Tuple[int, int]:
    """
    박스의 중심점 계산
    
    Args:
        box: 박스 객체
        
    Returns:
        (center_x, center_y) 튜플
    """
    center_x = box.x + box.width // 2
    center_y = box.y + box.height // 2
    return center_x, center_y


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

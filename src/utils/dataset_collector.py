"""
dataset_collector.py - 탐지 데이터 자동 수집 및 YOLO 라벨링
설명: 탐지 결과를 이미지 + YOLO 포맷 라벨로 자동 저장
"""

import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# 데이터셋 저장 디렉터리의 최소 여유 공간 (기본 500MB)
_MIN_FREE_DISK_BYTES = 500 * 1024 * 1024

logger = logging.getLogger(__name__)


@dataclass
class FrameMetadata:
    """프레임 메타데이터"""
    frame_id: int
    image_path: str
    camera_id: str
    timestamp: float
    frame_shape: Tuple[int, int, int]  # (H, W, C)
    detection_count: int
    class_distribution: Dict[str, int]  # 클래스별 개수


class DatasetCollector:
    """탐지 결과 자동 수집"""

    _ANNOTATE_COLORS = {
        'helmet_wearing': (0, 255, 0),
        'helmet_missing': (0, 0, 255),
        'fall_detected':  (255, 0, 255),
        'no_fall':        (255, 255, 0),
        'person':         (255, 255, 255),
    }
    _DEFAULT_ANNOTATE_COLOR = (128, 128, 128)
    
    def __init__(
        self,
        output_dir: str = './collected_data',
        format: str = 'yolo',  # 'yolo' or 'coco'
        save_images: bool = True,
        image_quality: int = 95
    ):
        """
        매개변수:
            output_dir: 저장 디렉터리
            format: 'yolo' (텍스트) 또는 'coco' (JSON)
            save_images: 이미지 파일 저장 여부
            image_quality: JPEG 품질 (1-100)
        """
        self.output_dir = Path(output_dir)
        self.format = format
        self.save_images = save_images
        self.image_quality = image_quality
        
        # 디렉터리 구조 생성
        self.images_dir   = self.output_dir / "images"
        self.labels_dir   = self.output_dir / "labels"
        self.metadata_dir = self.output_dir / "metadata"
        self.annotated_dir = self.output_dir / "annotated"

        for d in (self.images_dir, self.labels_dir, self.metadata_dir, self.annotated_dir):
            d.mkdir(parents=True, exist_ok=True)
        
        # 메타데이터 저장
        self.frame_metadata: List[FrameMetadata] = []
        self.frame_counter = 0
        self.class_name_to_id = {}  # 클래스명 -> ID 매핑
        self.id_to_class_name = {}  # ID -> 클래스명 매핑
        self._load_class_mapping()

    def _load_class_mapping(self):
        """classes.txt 로드 또는 생성"""
        classes_file = self.output_dir / "classes.txt"
        if classes_file.exists():
            with open(classes_file, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    class_name = line.strip()
                    self.class_name_to_id[class_name] = idx
                    self.id_to_class_name[idx] = class_name
        else:
            # 기본 클래스 설정 (events.py의 EventType 기준)
            default_classes = [
                'person',
                'helmet',
                'head',
                'fall_detected',
                'not_fall',
                'danger_zone',
                'unsafe_behavior'
            ]
            for idx, class_name in enumerate(default_classes):
                self.class_name_to_id[class_name] = idx
                self.id_to_class_name[idx] = class_name
            self._save_class_mapping(classes_file)

    def _save_class_mapping(self, file_path: Path):
        """클래스 매핑 저장"""
        with open(file_path, 'w', encoding='utf-8') as f:
            for idx in sorted(self.id_to_class_name.keys()):
                f.write(f"{self.id_to_class_name[idx]}\n")

    def register_class(self, class_name: str) -> int:
        """새로운 클래스 등록"""
        if class_name not in self.class_name_to_id:
            new_id = len(self.class_name_to_id)
            self.class_name_to_id[class_name] = new_id
            self.id_to_class_name[new_id] = class_name
            self._save_class_mapping(self.output_dir / "classes.txt")
        return self.class_name_to_id[class_name]

    def _check_disk_space(self) -> bool:
        """저장 디렉터리의 여유 공간이 최소 기준 이상인지 확인한다.

        Returns:
            여유 공간이 충분하면 True, 부족하면 False.
        """
        try:
            usage = shutil.disk_usage(self.output_dir)
            if usage.free < _MIN_FREE_DISK_BYTES:
                logger.warning(
                    "디스크 여유 공간 부족 (%.0fMB 남음, 최소 %.0fMB 필요) — 프레임 저장 건너뜀",
                    usage.free / 1024 / 1024,
                    _MIN_FREE_DISK_BYTES / 1024 / 1024,
                )
                return False
        except OSError as exc:
            logger.error("디스크 용량 확인 실패: %s", exc)
        return True

    def save_frame(
        self,
        frame: np.ndarray,
        detections: List,  # ai_analysis.DetectionEvent 리스트
        image_name: Optional[str] = None,
        camera_id: str = "unknown"
    ):
        """프레임 및 탐지 결과 저장
        
        매개변수:
            frame: 입력 이미지
            detections: 탐지 결과 리스트
            image_name: 이미지 파일명 (None이면 자동 생성)
            camera_id: 카메라 ID
        """
        if not self._check_disk_space():
            return

        if image_name is None:
            image_name = f"frame_{self.frame_counter:06d}.jpg"
        
        image_path = self.images_dir / image_name
        label_path = self.labels_dir / image_name.replace('.jpg', '.txt').replace('.png', '.txt')
        
        # 이미지 저장
        if self.save_images:
            cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, self.image_quality])
        
        h, w, c = frame.shape
        
        # 라벨 저장
        class_dist = {}
        self._save_yolo_label(label_path, detections, h, w, class_dist)
        
        # 박스가 그려진 이미지 저장 (시각적 확인용)
        self._save_annotated_image(frame.copy(), detections, image_name)
        
        # 메타데이터 기록
        metadata = FrameMetadata(
            frame_id=self.frame_counter,
            image_path=str(image_path.relative_to(self.output_dir)),
            camera_id=camera_id,
            timestamp=time.time(),
            frame_shape=(h, w, c),
            detection_count=len(detections),
            class_distribution=class_dist
        )
        self.frame_metadata.append(metadata)
        
        self.frame_counter += 1

    def _save_annotated_image(
        self,
        frame: np.ndarray,
        detections: List,
        image_name: str
    ):
        """박스가 그려진 이미지 저장 (시각적 확인용)"""
        annotated_path = self.annotated_dir / image_name
        
        # 각 탐지 결과에 박스 그리기
        for detection in detections:
            class_name = detection.event_type.value
            bbox = detection.to_dict().get('bbox', {})
            x, y, w, h = bbox.get('x', 0), bbox.get('y', 0), bbox.get('width', 0), bbox.get('height', 0)
            confidence = detection.confidence
            
            # 클래스별 색상
            color = self._ANNOTATE_COLORS.get(class_name.lower(), self._DEFAULT_ANNOTATE_COLOR)
            
            # 박스 그리기
            cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
            
            # 라벨 텍스트
            label_text = f"{class_name} {confidence:.2f}"
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            # 배경 박스
            cv2.rectangle(frame, (int(x), int(y) - label_size[1] - 10), 
                         (int(x) + label_size[0], int(y)), color, -1)
            
            # 텍스트
            cv2.putText(frame, label_text, (int(x), int(y) - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # 저장
        cv2.imwrite(str(annotated_path), frame, [cv2.IMWRITE_JPEG_QUALITY, self.image_quality])
    
    def _save_yolo_label(
        self,
        label_path: Path,
        detections: List,
        frame_h: int,
        frame_w: int,
        class_dist: Dict
    ):
        """YOLO 형식 라벨 저장
        
        YOLO 형식:
        <class_id> <x_center> <y_center> <width> <height>
        (모든 좌표는 0~1 사이의 정규화된 값)
        """
        lines = []
        for detection in detections:
            class_name = detection.event_type.value
            class_id = self.register_class(class_name)
            
            # 클래스 분포 업데이트
            class_dist[class_name] = class_dist.get(class_name, 0) + 1
            
            bbox = detection.to_dict().get('bbox', {})
            x, y, w, h = bbox.get('x', 0), bbox.get('y', 0), bbox.get('width', 0), bbox.get('height', 0)
            
            # YOLO 형식: x,y는 중심점 좌표
            x_center = x + w / 2
            y_center = y + h / 2
            
            # 정규화 (0~1)
            x_norm = x_center / frame_w if frame_w > 0 else 0
            y_norm = y_center / frame_h if frame_h > 0 else 0
            w_norm = w / frame_w if frame_w > 0 else 0
            h_norm = h / frame_h if frame_h > 0 else 0
            
            lines.append(f"{class_id} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}")
        
        with open(label_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n' if lines else '')

    def export_coco(self, output_file: str = 'annotations.json'):
        """COCO 형식 JSON 내보내기
        
        COCO 형식: {images: [...], annotations: [...], categories: [...]}
        """
        coco_data = {
            'images': [],
            'annotations': [],
            'categories': [
                {'id': idx, 'name': self.id_to_class_name[idx]}
                for idx in sorted(self.id_to_class_name.keys())
            ]
        }
        
        annotation_id = 0
        for frame_meta in self.frame_metadata:
            # 이미지 항목
            image_item = {
                'id': frame_meta.frame_id,
                'file_name': frame_meta.image_path,
                'height': frame_meta.frame_shape[0],
                'width': frame_meta.frame_shape[1],
                'camera_id': frame_meta.camera_id,
                'timestamp': frame_meta.timestamp
            }
            coco_data['images'].append(image_item)
            
            # 라벨 파일에서 어노테이션 읽기 및 추가
            label_file = self.labels_dir / Path(frame_meta.image_path).stem
            label_file = label_file.with_suffix('.txt')
            
            if label_file.exists():
                with open(label_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_norm, y_norm, w_norm, h_norm = map(float, parts[1:5])
                            
                            h, w = frame_meta.frame_shape[0], frame_meta.frame_shape[1]
                            x = int(x_norm * w)
                            y = int(y_norm * h)
                            width = int(w_norm * w)
                            height = int(h_norm * h)
                            
                            annotation = {
                                'id': annotation_id,
                                'image_id': frame_meta.frame_id,
                                'category_id': class_id,
                                'bbox': [x, y, width, height],
                                'area': width * height,
                                'iscrowd': 0
                            }
                            coco_data['annotations'].append(annotation)
                            annotation_id += 1
        
        output_path = self.output_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2, ensure_ascii=False)
        
        logger.info("COCO 형식 내보내기: %s", output_path)

    def get_statistics(self) -> Dict:
        """수집 통계 반환"""
        total_detections = sum(m.detection_count for m in self.frame_metadata)
        class_dist_total = {}
        for m in self.frame_metadata:
            for class_name, count in m.class_distribution.items():
                class_dist_total[class_name] = class_dist_total.get(class_name, 0) + count
        
        return {
            'total_frames': len(self.frame_metadata),
            'total_detections': total_detections,
            'class_distribution': class_dist_total,
            'output_dir': str(self.output_dir)
        }

    def print_statistics(self):
        """통계 출력"""
        stats = self.get_statistics()
        logger.info(
            "\n%s\n\ud0d0\uc9c0 \ub370\uc774\ud130\uc14b \uc218\uc9d1 \ud1b5\uacc4\n%s\n"
            "\uc804\uccb4 \ud504\ub808\uc784: %s\n"
            "\uc804\uccb4 \ud0d0\uc9c0: %s\n"
            "\ud074\ub798\uc2a4 \ubd84\ud3ec: %s\n"
            "\uc800\uc7a5 \uc704\uce58: %s\n%s",
            "=" * 60, "=" * 60,
            stats['total_frames'],
            stats['total_detections'],
            stats['class_distribution'],
            stats['output_dir'],
            "=" * 60,
        )


__all__ = ["DatasetCollector", "FrameMetadata"]

import requests
import json
import logging
from typing import Union, Dict, Optional
from dataclasses import dataclass
from ..core.events import DetectionEvent
from ..config import default_config

logger = logging.getLogger(__name__)

# 서버 통신 설정 상수 (하위 호환성)
DEFAULT_TIMEOUT = 5  # 기본 타임아웃 (초)
DEFAULT_RETRY_COUNT = 3  # 기본 재시도 횟수
RETRY_DELAY = 1  # 재시도 간격 (초)


@dataclass
class ServerResponse:
    """서버 응답 래퍼"""
    success: bool
    status_code: Optional[int] = None
    data: Optional[Dict] = None
    error_message: Optional[str] = None


def send_event(
    event: Union[Dict, 'DetectionEvent'],
    server_url: Optional[str] = None,
    timeout: Optional[int] = None,
    retry_count: Optional[int] = None
) -> ServerResponse:
    """
    이벤트를 서버로 전송 (JSON 형식, 재시도 로직 포함)
    
    Args:
        event: 이벤트 딕셔너리 또는 DetectionEvent 객체
        server_url: 서버 URL (None이면 config에서 가져옴)
        timeout: 요청 타임아웃 (초, None이면 config에서 가져옴)
        retry_count: 실패 시 재시도 횟수 (None이면 config에서 가져옴)
        
    Returns:
        ServerResponse: 전송 결과 객체
    
    예시:
        # config 사용 (권장)
        result = send_event(event)
        
        # 직접 지정 (하위 호환성)
        result = send_event(event, server_url="http://...", retry_count=3)
        
        if result.success:
            print("전송 성공!")
    """
    # config에서 기본값 가져오기
    if server_url is None:
        server_url = default_config.server.url
    if timeout is None:
        timeout = default_config.server.timeout
    if retry_count is None:
        retry_count = default_config.server.retry_count
    import time
    
    # DetectionEvent 객체이면 딕셔너리로 변환
    if hasattr(event, 'to_dict'):
        event_data = event.to_dict()
    else:
        event_data = event
    
    # JSON 직렬화 테스트
    try:
        json_payload = json.dumps(event_data, ensure_ascii=False, indent=2)
    except (TypeError, ValueError) as e:
        logger.error(f"❌ JSON 변환 오류: {e}")
        return ServerResponse(success=False, error_message=f"JSON 변환 실패: {e}")
    
    logger.info(f"[SERVER SEND] Sending event to {server_url}")
    logger.debug(f"[SERVER PAYLOAD]\n{json_payload}")
    
    # 재시도 로직
    last_error = None
    for attempt in range(retry_count):
        try:
            # HTTP POST 요청
            response = requests.post(
                server_url,
                json=event_data,
                headers={"Content-Type": "application/json"},
                timeout=timeout
            )
            
            # 상태 코드 확인
            if response.status_code in [200, 201]:
                logger.info(f"✅ 전송 성공: {response.status_code}")
                response_data = response.json() if response.text else None
                return ServerResponse(
                    success=True,
                    status_code=response.status_code,
                    data=response_data
                )
            else:
                error_msg = f"서버 응답 오류: {response.status_code}"
                if response.status_code == 405:
                    error_msg = "405 Method Not Allowed: 서버가 POST를 허용하지 않습니다"
                
                logger.warning(f"⚠️ {error_msg}")
                last_error = error_msg
                
                # 재시도 전 대기
                if attempt < retry_count - 1:
                    logger.info(f"재시도 {attempt + 1}/{retry_count} (⏳ {RETRY_DELAY}초 후)")
                    time.sleep(RETRY_DELAY)
                    continue
                
                return ServerResponse(
                    success=False,
                    status_code=response.status_code,
                    error_message=error_msg
                )
                
        except requests.exceptions.ConnectionError as e:
            logger.error(f"❌ 연결 실패: 서버({server_url})에 연결할 수 없습니다")
            last_error = f"연결 실패: {e}"
            
            if attempt < retry_count - 1:
                logger.info(f"재시도 {attempt + 1}/{retry_count} (⏳ {RETRY_DELAY}초 후)")
                time.sleep(RETRY_DELAY)
                continue
            
        except requests.exceptions.Timeout as e:
            logger.error(f"❌ 타임아웃: 요청이 {server_url}에서 시간 초과 ({timeout}초)")
            last_error = f"타임아웃: {e}"
            
            if attempt < retry_count - 1:
                logger.info(f"재시도 {attempt + 1}/{retry_count} (⏳ {RETRY_DELAY}초 후)")
                time.sleep(RETRY_DELAY)
                continue
            
        except Exception as e:
            logger.error(f"❌ 전송 오류: {e}")
            last_error = str(e)
            
            if attempt < retry_count - 1:
                logger.info(f"재시도 {attempt + 1}/{retry_count} (⏳ {RETRY_DELAY}초 후)")
                time.sleep(RETRY_DELAY)
                continue
    
    # 모든 재시도 실패
    logger.error(f"❌ {retry_count}회 재시도 후 전송 실패")
    print(f"[SERVER] 서버 전송 실패. 로컬로 저장됩니다.")
    _save_event_locally(event_data)
    
    return ServerResponse(success=False, error_message=last_error)


def _save_event_locally(event_data):
    """서버 전송 실패 시 이벤트를 로컬 JSON 파일로 저장"""
    import os
    from pathlib import Path
    from datetime import datetime
    
    try:
        log_dir = "event_logs"
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{log_dir}/event_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(event_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 이벤트 로컬 저장: {filename}")
    except Exception as e:
        logger.error(f"로컬 저장 실패: {e}")

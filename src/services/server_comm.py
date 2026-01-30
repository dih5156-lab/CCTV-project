import requests
import json
import logging
from typing import Union, Dict, Optional
from dataclasses import dataclass
from ..core.events import DetectionEvent
from ..config import default_config

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 5
DEFAULT_RETRY_COUNT = 3
RETRY_DELAY = 1


@dataclass
class ServerResponse:
    """서버 응답 래퍼"""
    success: bool
    status_code: Optional[int] = None
    data: Optional[Dict] = None
    error_message: Optional[str] = None
    
    def __bool__(self):
        """응답의 bool 평가 허용"""
        return self.success


def send_event(
    event: Union[Dict, 'DetectionEvent'],
    server_url: Optional[str] = None,
    timeout: Optional[int] = None,
    retry_count: Optional[int] = None
) -> ServerResponse:
    """
    서버로 이벤트 전송 (JSON 형식, 재시도 로직 포함)
    
    Args:
        event: 이벤트 딕셔너리 또는 DetectionEvent 객체
        server_url: 서버 URL (None이면 config 사용)
        timeout: 요청 타임아웃 (초, None이면 config 사용)
        retry_count: 실패 시 재시도 횟수 (None이면 config 사용)
        
    Returns:
        ServerResponse: 전송 결과 객체
    
    Example:
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
    
    # 매개변수 검증
    if not server_url:
        error_msg = "서버 URL이 설정되지 않음"
        logger.error(error_msg)
        return ServerResponse(success=False, error_message=error_msg)
    
    if not server_url.startswith(('http://', 'https://')):
        error_msg = f"잘못된 서버 URL: {server_url} (http:// 또는 https://로 시작해야 함)"
        logger.error(error_msg)
        return ServerResponse(success=False, error_message=error_msg)
    
    if timeout <= 0:
        error_msg = f"잘못된 timeout: {timeout} (양수여야 함)"
        logger.error(error_msg)
        return ServerResponse(success=False, error_message=error_msg)
    
    if retry_count < 0:
        error_msg = f"잘못된 retry_count: {retry_count} (음수가 아니어야 함)"
        logger.error(error_msg)
        return ServerResponse(success=False, error_message=error_msg)
    
    import time
    
    # DetectionEvent 객체를 딕셔너리로 변환
    if hasattr(event, 'to_dict'):
        event_data = event.to_dict()
    else:
        event_data = event
    
    # event_data가 딕셔너리인지 검증
    if not isinstance(event_data, dict):
        error_msg = f"잘못된 이벤트 데이터 타입: {type(event_data).__name__} (dict 필요)"
        logger.error(error_msg)
        return ServerResponse(success=False, error_message=error_msg)
    
    try:
        json_payload = json.dumps(event_data, ensure_ascii=False, indent=2)
    except (TypeError, ValueError) as e:
        logger.error(f"JSON 변환 오류: {e}")
        return ServerResponse(success=False, error_message=f"JSON 변환 실패: {e}")
    
    logger.info(f"[서버 전송] {server_url}로 이벤트 전송 중")
    logger.debug(f"[서버 페이로드]\n{json_payload}")
    
    last_error = None
    for attempt in range(retry_count):
        try:
            response = requests.post(
                server_url,
                json=event_data,
                headers={"Content-Type": "application/json"},
                timeout=timeout
            )
            
            if response.status_code in [200, 201]:
                logger.info(f"전송 성공: {response.status_code}")
                response_data = response.json() if response.text else None
                return ServerResponse(
                    success=True,
                    status_code=response.status_code,
                    data=response_data
                )
            else:
                error_msg = f"서버 응답 오류: {response.status_code}"
                if response.status_code == 405:
                    error_msg = "405 Method Not Allowed: 서버가 POST를 허용하지 않음"
                
                logger.warning(f"WARN: {error_msg}")
                last_error = error_msg
                
                if attempt < retry_count - 1:
                    logger.info(f"재시도 {attempt + 1}/{retry_count} ({RETRY_DELAY}초 대기)")
                    time.sleep(RETRY_DELAY)
                    continue
                
                return ServerResponse(
                    success=False,
                    status_code=response.status_code,
                    error_message=error_msg
                )
                
        except requests.exceptions.ConnectionError as e:
            logger.error(f"연결 실패: 서버에 연결할 수 없음 ({server_url})")
            last_error = f"연결 실패: {e}"
            
            if attempt < retry_count - 1:
                logger.info(f"재시도 {attempt + 1}/{retry_count} ({RETRY_DELAY}초 대기)")
                time.sleep(RETRY_DELAY)
                continue
            
        except requests.exceptions.Timeout as e:
            logger.error(f"타임아웃: {server_url}로의 요청 타임아웃 ({timeout}초)")
            last_error = f"타임아웃: {e}"
            
            if attempt < retry_count - 1:
                logger.info(f"재시도 {attempt + 1}/{retry_count} ({RETRY_DELAY}초 대기)")
                time.sleep(RETRY_DELAY)
                continue
            
        except Exception as e:
            logger.error(f"전송 오류: {e}")
            last_error = str(e)
            
            if attempt < retry_count - 1:
                logger.info(f"재시도 {attempt + 1}/{retry_count} ({RETRY_DELAY}초 대기)")
                time.sleep(RETRY_DELAY)
                continue
    
    logger.error(f"{retry_count}번 재시도 후 전송 실패")
    print(f"[서버] 서버 전송 실패. 로컬에 저장합니다.")
    _save_event_locally(event_data)
    
    return ServerResponse(success=False, error_message=last_error)


def _save_event_locally(event_data):
    """서버 전송 실패 시 이벤트를 로컬 JSON 파일로 저장"""
    import os
    from pathlib import Path
    from datetime import datetime
    
    try:
        log_dir = Path("event_logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = log_dir / f"event_{timestamp}.json"
        
        # 유효한 JSON 데이터인지 확인
        if not isinstance(event_data, dict):
            logger.error(f"dict가 아닌 이벤트 데이터는 저장할 수 없음: {type(event_data)}")
            return
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(event_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"이벤트가 로컬에 저장됨: {filename}")
    except OSError as e:
        logger.error(f"로컬 저장 중 파일 시스템 오류: {e}")
    except Exception as e:
        logger.error(f"로컬 저장 실패: {e}")

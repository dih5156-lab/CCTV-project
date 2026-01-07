# -*- coding: utf-8 -*-
"""
[test_server.py] 이벤트 수신 테스트 서버
제작일 : 2025-11-19
설명: 카메라에서 전송한 이벤트를 수신하는 Flask 서버

사용법:
    python test_server.py
    
기본 포트: 8000
엔드포인트: http://localhost:8000/api/events
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from flask import Flask, request, jsonify
try:
    from flask_cors import CORS
    has_cors = True
except Exception:
    has_cors = False
import json
import logging
from datetime import datetime

app = Flask(__name__)
if has_cors:
    CORS(app)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 이벤트 저장소
events_received = []


@app.route('/api/events', methods=['GET', 'POST'])
def receive_event():
    """
    이벤트 수신 엔드포인트
    카메라에서 POST로 전송한 JSON 이벤트를 수신합니다.
    """
    try:
        # GET 요청: 간단한 안내 반환 (브라우저 테스트용)
        if request.method == 'GET':
            return jsonify({
                "message": "POST events as JSON to this endpoint. Use /api/events/stats to view received events."
            }), 200

        # JSON 파싱
        event_data = request.get_json()
        
        if not event_data:
            return jsonify({"error": "No JSON data"}), 400
        
        # 수신 시간 추가
        event_data['received_at'] = datetime.now().isoformat()
        
        # 이벤트 저장
        events_received.append(event_data)
        
        # 콘솔 출력
        print("\n" + "="*60)
        print("✅ 이벤트 수신!")
        print("="*60)
        print(json.dumps(event_data, ensure_ascii=False, indent=2))
        print("="*60 + "\n")
        
        # 로깅
        logger.info(f"이벤트 수신: {event_data.get('type')} (신뢰도: {event_data.get('confidence')})")
        
        # 응답
        return jsonify({
            "status": "success",
            "message": "Event received successfully",
            "event_id": len(events_received)
        }), 200
    
    except Exception as e:
        logger.error(f"❌ 이벤트 수신 오류: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/events/stats', methods=['GET'])
def get_stats():
    """
    이벤트 통계 조회
    지금까지 수신한 이벤트 통계를 반환합니다.
    """
    event_types = {}
    for event in events_received:
        event_type = event.get('type', 'unknown')
        event_types[event_type] = event_types.get(event_type, 0) + 1
    
    return jsonify({
        "total_events": len(events_received),
        "event_types": event_types,
        "events": events_received[-10:]  # 최근 10개만
    }), 200


@app.route('/api/events/all', methods=['GET'])
def get_all_events():
    """
    모든 이벤트 조회
    """
    return jsonify({
        "total": len(events_received),
        "events": events_received
    }), 200


@app.route('/api/events/clear', methods=['DELETE'])
def clear_events():
    """
    이벤트 초기화
    """
    global events_received
    count = len(events_received)
    events_received = []
    return jsonify({
        "message": f"Cleared {count} events"
    }), 200


@app.route('/health', methods=['GET'])
def health():
    """
    서버 상태 확인
    """
    return jsonify({"status": "ok", "message": "Server is running"}), 200


@app.route('/', methods=['GET'])
def index():
    """
    루트 엔드포인트 (서버 정보)
    """
    return jsonify({
        "server": "Event Receiver Server",
        "version": "1.0",
        "endpoints": {
            "POST /api/events": "이벤트 수신",
            "GET /api/events/stats": "이벤트 통계",
            "GET /api/events/all": "모든 이벤트 조회",
            "DELETE /api/events/clear": "이벤트 초기화",
            "GET /health": "서버 상태 확인"
        }
    }), 200


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 이벤트 수신 서버 시작")
    print("="*60)
    print("📍 서버 주소: http://localhost:8000")
    print("📊 통계 조회: http://localhost:8000/api/events/stats")
    print("🔄 모든 이벤트: http://localhost:8000/api/events/all")
    print("💾 이벤트 초기화: DELETE http://localhost:8000/api/events/clear")
    print("✅ 건강 상태: http://localhost:8000/health")
    print("="*60)
    print("⚠️  주의: 이 서버를 실행한 상태에서 camera_inference.py를 실행하세요\n")
    
    app.run(host='0.0.0.0', port=8000, debug=False)

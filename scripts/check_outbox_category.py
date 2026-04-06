"""임시 동작 확인 스크립트 — 실행 후 삭제해도 됩니다."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.edgex.device_service import CCTVDeviceService

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sample_outbox.db")

# 기존 파일 초기화
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

svc = CCTVDeviceService({
    "coreMetadataUrl": "http://localhost:59881",
    "coreDataUrl": "http://localhost:59880",
    "deviceServiceName": "test",
    "enableStoreAndForward": True,
    "outboxDbPath": DB_PATH,
})
svc._init_outbox()

svc._store_failed_detection_event("cam1", {"type": "helmet"},       "network error")
svc._store_failed_detection_event("cam1", {"type": "fall_detected"},"network error")
svc._store_failed_detection_event("cam2", {"type": "intrusion"},    "network error")
svc._store_failed_detection_event("cam2", {"type": "person"},       "network error")
svc._store_failed_detection_event("cam3", {"type": "face_recognized"}, "network error")
svc._store_failed_detection_event("cam3", {"type": "danger_zone"},  "network error")

all_rows    = svc.get_pending_detection_events()
person_rows = svc.get_pending_detection_events(data_category="person")
camera_rows = svc.get_pending_detection_events(data_category="camera")

print("=== 전체 pending ===")
for r in all_rows:
    etype    = r["event_data"].get("type", "?")
    category = r["data_category"]
    cam      = r["camera_id"]
    print(f"  id={r['id']}  camera={cam:<5}  type={etype:<15}  category={category}")

print()
print(f"person 카테고리: {len(person_rows)}건")
print(f"camera 카테고리: {len(camera_rows)}건")
print()
print(f"DB 파일 위치: {DB_PATH}")
print("VS Code에서 위 파일을 열면 SQLite Viewer로 바로 확인할 수 있습니다.")

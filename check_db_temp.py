import sqlite3, os

db = "/data/edgex_outbox.db"
if not os.path.exists(db):
    print("DB 파일 없음")
else:
    conn = sqlite3.connect(db)
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    print(f"테이블 목록: {tables}")

    if "detection_outbox" in tables:
        cctv_total = conn.execute("SELECT COUNT(*) FROM detection_outbox").fetchone()[0]
        cctv_by_status = conn.execute("SELECT status, COUNT(*) FROM detection_outbox GROUP BY status").fetchall()
        cctv_by_category = conn.execute("SELECT data_category, status, COUNT(*) FROM detection_outbox GROUP BY data_category, status").fetchall()
        latest = conn.execute(
            "SELECT camera_id, data_category, status, datetime(created_at) FROM detection_outbox ORDER BY id DESC LIMIT 5"
        ).fetchall()
        print(f"\n[CCTV detection_outbox] 총: {cctv_total}")
        print(f"  상태별: {cctv_by_status}")
        print(f"  카테고리별: {cctv_by_category}")
        print("  최근 5건:")
        for r in latest:
            print("   ", r)

    sensor_total = conn.execute("SELECT COUNT(*) FROM edgex_outbox").fetchone()[0]
    sensor_by_status = conn.execute("SELECT status, COUNT(*) FROM edgex_outbox GROUP BY status").fetchall()
    print(f"\n[센서 edgex_outbox] 총: {sensor_total}, 상태별: {sensor_by_status}")
    conn.close()

"""
main.py
=======
Go 원본: aiot-tlv-parser/main.go

AIoT TLV Parser 서비스 진입점입니다.
Go의 gin HTTP 서버 + goroutine graceful shutdown
→ Python의 Flask HTTP 서버 + signal 핸들러로 변환되었습니다.

초기화 순서:
  1. .env 파일 로드 (python-dotenv)
  2. Config 로드 및 출력
  3. DB 커넥션 풀 초기화
  4. SensorService 초기화
  5. BatchManager 초기화 및 시작
  6. 디바이스 목록 갱신 완료 대기 (최대 63초)
  7. MQTT Manager 초기화
  8. Flask HTTP 서버 시작
  9. 종료 시그널 처리 (graceful shutdown)

의존성:
  pip install flask python-dotenv psycopg2-binary paho-mqtt
"""

import logging
import signal
import threading
import time

# 로깅 기본 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    # ──────────────────────────────────────────────
    # 1. 환경 변수 로드 (.env 파일)
    # Go: if err := godotenv.Load(); err != nil { log.Println("No .env file found") }
    # ──────────────────────────────────────────────
    try:
        from dotenv import load_dotenv
        if not load_dotenv():
            logger.info("No .env file found")
    except ImportError:
        logger.warning("python-dotenv not installed. Skipping .env load.")

    # ──────────────────────────────────────────────
    # 2. 설정 로드
    # Go: cfg := config.Load(); cfg.PrintConfig()
    # ──────────────────────────────────────────────
    from config.config import load as load_config
    cfg = load_config()
    cfg.print_config()

    # ──────────────────────────────────────────────
    # 3. 데이터베이스 연결 풀 초기화
    # Go: db, err := database.Init(cfg.Database)
    # ──────────────────────────────────────────────
    from database.connection import init as db_init
    db = db_init(cfg.database)

    def shutdown_db():
        db.close()  # Go: defer db.Close()

    # ──────────────────────────────────────────────
    # 4. 서비스 초기화
    # Go: sensorService := service.NewSensorService(db)
    # ──────────────────────────────────────────────
    from mqtt.edgex_forwarder import create_from_env as create_edgex_forwarder
    from service.sensor_service import SensorService
    edgex_forwarder = create_edgex_forwarder()
    sensor_service = SensorService(db, edgex_forwarder=edgex_forwarder)

    # ──────────────────────────────────────────────
    # 6. 배치 매니저 생성 및 초기화
    # Go: batchManager := batch.NewBatchManager(...)
    # ──────────────────────────────────────────────
    from batch.manager import BatchManager
    batch_manager = BatchManager(
        cfg=cfg.batch,
        device_update_callback=sensor_service.device_info.update_devices_from_batch,
    )
    try:
        batch_manager.init()
    except Exception as e:
        logger.warning(f"Failed to initialize batch manager: {e}")

    # ──────────────────────────────────────────────
    # 7. 디바이스 목록 갱신 완료 대기 (최대 63초)
    # Go: for i := 0; i < 21; i++ { time.Sleep(3s); if currentCount > 0 { break } }
    # ──────────────────────────────────────────────
    for i in range(21):
        time.sleep(3)
        count = sensor_service.device_info.get_device_count()
        if count > 0:
            logger.info(f"Device list updated successfully. Total devices: {count}")
            break
        if i == 20:
            logger.warning("Device list update timeout(1 minutes). Using initial devices: 0")

    # ──────────────────────────────────────────────
    # 9. MQTT 클라이언트 초기화
    # Go: mqttManager := mqtt.NewManager(cfg.MQTT)
    # ──────────────────────────────────────────────
    from mqtt.manager import Manager as MQTTManager
    mqtt_manager = MQTTManager(cfg.mqtt)
    try:
        mqtt_manager.init(cfg.mqtt, sensor_service)
    except Exception as e:
        logger.warning(f"Failed to initialize MQTT clients: {e}")

    # ──────────────────────────────────────────────
    # 11. HTTP 서버 설정 (Flask → Go의 gin 대응)
    # Go: router := gin.Default()
    # ──────────────────────────────────────────────
    try:
        from flask import Flask, jsonify
        app = Flask(__name__)

        # CORS 미들웨어
        # Go: router.Use(func(c *gin.Context) { c.Header("Access-Control-Allow-Origin", "*"); ... })
        @app.after_request
        def add_cors_headers(response):
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
            return response

        # Go: router.GET("/health", func(c *gin.Context) { ... })
        @app.route("/health")
        def health():
            return jsonify({
                "status": "healthy",
                "timestamp": int(time.time()),
            })

        # Go: router.GET("/", func(c *gin.Context) { ... })
        @app.route("/")
        def index():
            return jsonify({
                "message": "AIoT TLV Parser Service",
                "app_version": cfg.server.app_version,
                "timestamp": int(time.time()),
            })

        flask_available = True
    except ImportError:
        logger.warning("Flask not installed. HTTP server disabled.")
        flask_available = False

    # ──────────────────────────────────────────────
    # 12. HTTP 서버 시작 (별도 스레드)
    # Go: go func() { srv.ListenAndServe() }()
    # ──────────────────────────────────────────────
    port = cfg.server.port or "3500"
    server_thread = None

    if flask_available:
        def run_server():
            logger.info(f"서버가 {port}번 포트에서 실행 중입니다.")
            app.run(host="0.0.0.0", port=int(port), use_reloader=False)

        server_thread = threading.Thread(target=run_server, daemon=True, name="HTTPServer")
        server_thread.start()

    # ──────────────────────────────────────────────
    # Graceful shutdown 처리
    # Go: signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM); <-quit
    # ──────────────────────────────────────────────
    shutdown_event = threading.Event()

    def handle_signal(signum, frame):
        logger.info("Shutting down server...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # 종료 대기 (Go: <-quit)
    shutdown_event.wait()

    # 정리 (Go: defer 순서대로)
    logger.info("Stopping MQTT connections...")
    mqtt_manager.disconnect_all()

    logger.info("Stopping batch jobs...")
    batch_manager.stop_all()

    logger.info("Closing sensor service...")
    sensor_service.close()

    logger.info("Closing database...")
    shutdown_db()

    logger.info("Server exited")


if __name__ == "__main__":
    main()

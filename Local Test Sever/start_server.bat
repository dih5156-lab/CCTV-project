@echo off
REM [start_server.bat] 서버 자동 시작 스크립트
REM 설명: test_server.py를 새 창에서 자동으로 실행

echo.
echo ======================================
echo   테스트 서버 자동 시작
echo ======================================
echo.

echo [1/3] 필수 패키지 확인 중...
python -c "import requests, flask, ultralytics, cv2, numpy" >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ 패키지가 설치되지 않았습니다!
    echo.
    echo 다음 명령어를 실행하세요:
    echo   pip install requests flask ultralytics opencv-python numpy
    echo.
    pause
    exit /b 1
)
echo ✅ 패키지 확인 완료

echo.
echo [2/3] 테스트 서버 시작...
start "Event Server" python test_server.py

echo ✅ 서버가 새 창에서 실행 중입니다

echo.
echo [3/3] 2초 대기 중...
timeout /t 2 /nobreak

echo.
echo ======================================
echo   ✅ 서버 시작 완료!
echo ======================================
echo.
echo 📍 서버 주소: http://localhost:8000
echo.
echo 다음 명령어로 카메라 추론을 실행하세요:
echo   python camera_inference.py
echo.
echo 또는 다음 명령어로 진단을 실행하세요:
echo   python server_diagnostics.py
echo.
pause

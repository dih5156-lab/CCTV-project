@echo off
REM ========================================
REM  2-in-1 실행 스크립트: 서버 + 메인 동시 시작
REM  사용법: 이 파일 더블클릭 또는 cmd에서 실행
REM ========================================

setlocal enabledelayedexpansion

echo.
echo ========================================
echo   CCTV 객체탐지 시스템 시작
echo ========================================
echo.
echo [1/2] 테스트 서버 시작 중 (포트 8000)...
echo.

REM PowerShell에서 백그라운드로 test_server.py 실행
powershell -Command "Start-Process python -ArgumentList 'test_server.py' -WindowStyle Minimized"

REM 서버 시작 대기 (3초)
timeout /t 3 /nobreak

REM 연결 확인
python -c "import socket; s=socket.socket(); s.settimeout(1); print('연결 확인 중...' if s.connect_ex(('localhost', 8000))==0 else '서버 시작 실패')"

echo.
echo [2/2] 메인 프로그램 시작...
echo.

REM 메인 프로그램 실행 (사용자 옵션)
python main.py --mode single --display --zone-detection --collect-dataset

pause

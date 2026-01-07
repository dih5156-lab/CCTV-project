# ========================================
# 2-in-1 실행 스크립트 (PowerShell)
# 사용법: powershell -ExecutionPolicy Bypass -File start_all.ps1
# ========================================

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  CCTV 객체탐지 시스템 시작" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 1. test_server.py 백그라운드 시작
Write-Host "[1/2] 테스트 서버 시작 중 (포트 8000)..." -ForegroundColor Yellow
Start-Process python -ArgumentList "test_server.py" -WindowStyle Minimized -NoNewWindow

# 서버 시작 대기
Write-Host "서버 시작 대기 중 (3초)..." -ForegroundColor Gray
Start-Sleep -Seconds 3

# 연결 확인
Write-Host "연결 상태 확인..." -ForegroundColor Gray
$socketTest = @"
import socket
s = socket.socket()
s.settimeout(1)
if s.connect_ex(('localhost', 8000)) == 0:
    print('✅ 서버 정상')
else:
    print('❌ 서버 미응답')
s.close()
"@

python -c $socketTest

Write-Host ""
Write-Host "[2/2] 메인 프로그램 시작..." -ForegroundColor Yellow
Write-Host ""

# 2. 메인 프로그램 실행
python main.py --mode single --display --zone-detection --collect-dataset

Write-Host ""
Write-Host "프로그램 종료" -ForegroundColor Yellow

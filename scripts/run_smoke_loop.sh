#!/usr/bin/env bash
# run_smoke_loop.sh — smoke_test_data_flow.py를 지정 시간 동안 반복 실행
# 사용: ./scripts/run_smoke_loop.sh [실행시간(분)] [간격(초)]
# 예시: ./scripts/run_smoke_loop.sh 60 30   → 60분간 30초 간격으로 실행

set -euo pipefail

DURATION_MIN=${1:-30}
INTERVAL_SEC=${2:-30}
DURATION_SEC=$(( DURATION_MIN * 60 ))

PASS=0
FAIL=0
RUN=0
START=$(date +%s)
END=$(( START + DURATION_SEC ))

echo "=== CCTV 스모크 루프 시작 ==="
echo "  실행 시간: ${DURATION_MIN}분 / 간격: ${INTERVAL_SEC}초"
echo "  시작: $(date '+%Y-%m-%d %H:%M:%S')"
echo "==============================="

while [[ $(date +%s) -lt $END ]]; do
    RUN=$(( RUN + 1 ))
    TS=$(date '+%H:%M:%S')

    RESULT=$(python scripts/smoke_test_data_flow.py 2>&1)
    PASSED=$(echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('passed','?'))" 2>/dev/null || echo "error")

    if [[ "$PASSED" == "True" ]]; then
        PASS=$(( PASS + 1 ))
        echo "[$TS] #${RUN}  ✓ PASS  (누적 PASS=${PASS} FAIL=${FAIL})"
    else
        FAIL=$(( FAIL + 1 ))
        echo "[$TS] #${RUN}  ✗ FAIL  (누적 PASS=${PASS} FAIL=${FAIL})"
        # 실패 시 상세 출력
        echo "$RESULT" | python3 -c "
import sys, json
d = json.load(sys.stdin)
for c in d.get('checks', []):
    if not c.get('passed'):
        print(f\"  -> FAIL: {c['name']} | status={c['status']} | {c.get('detail','')[:200]}\")
" 2>/dev/null || echo "$RESULT" | tail -20
    fi

    ELAPSED=$(( $(date +%s) - START ))
    REMAINING=$(( END - $(date +%s) ))

    if [[ $REMAINING -le 0 ]]; then
        break
    fi

    # 다음 인터벌까지 대기 (남은 시간보다 길면 그냥 종료)
    SLEEP=$(( INTERVAL_SEC < REMAINING ? INTERVAL_SEC : REMAINING ))
    sleep "$SLEEP"
done

echo ""
echo "=== 루프 완료 ==="
echo "  총 실행: ${RUN}회 | PASS: ${PASS} | FAIL: ${FAIL}"
ELAPSED=$(( $(date +%s) - START ))
echo "  경과 시간: $(( ELAPSED / 60 ))분 $(( ELAPSED % 60 ))초"
if [[ $RUN -gt 0 ]]; then
    FAIL_RATE=$(( FAIL * 100 / RUN ))
    echo "  실패율: ${FAIL_RATE}%"
fi
echo "  종료: $(date '+%Y-%m-%d %H:%M:%S')"

[[ $FAIL -eq 0 ]] && exit 0 || exit 1

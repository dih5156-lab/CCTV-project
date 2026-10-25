"""전송 방식과 무관한 원본 저장 ACK 확인과 노드별 단일 송신 소유권."""

from __future__ import annotations

import fcntl
import os
import stat


def acquire_delivery_lock(path):
    """MQTT와 DDS가 같은 원본 큐를 동시에 전송하지 않도록 프로세스 락을 잡는다."""
    fd = os.open(
        str(path) + ".delivery.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
    )
    try:
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_mode & 0o077
        ):
            raise ValueError("delivery lock requires owner-only regular file")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        os.close(fd)
        raise
    return fd


def deliver_pending(context, node_id, controller, publish, stop):
    """발행 성공과 무관하게 중앙의 동일 ID/해시 저장 확인 뒤에만 원본을 완료한다."""
    # 중단 적체가 정상 유입에 밀리지 않도록 큐의 기존 상한까지 처리하고 쉰다.
    for item in context.pending(limit=100):
        if stop.is_set():
            return
        event_id = item["event_id"]
        try:
            if item["payload"]["edge"]["node_id"] != node_id:
                raise ValueError("wrong_node_result_context")
            code, receipt = controller.request(
                "GET", f"/v1/nodes/{node_id}/receipts/{event_id}"
            )
            if code == 200:
                if (
                    not isinstance(receipt, dict)
                    or receipt.get("event_id") != event_id
                    or receipt.get("sha256") != item["sha256"]
                ):
                    raise ValueError("central_receipt_mismatch")
                context.confirm(receipt)
            elif code == 404:
                context.record_error(event_id, publish(item["payload"]))
            else:
                context.record_error(event_id, f"central_http_{code}")
        except Exception as error:
            context.record_error(event_id, type(error).__name__)

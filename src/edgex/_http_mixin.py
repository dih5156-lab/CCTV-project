"""
EdgeX HTTP 통신 믹스인

EdgeX Foundry REST API 와의 통신에 필요한 모든 HTTP 유틸리티를 담당한다.
버전 폴백(v3 → v2 → v1), 헬스 프로브, 엔터티 CRUD 등을 포함한다.
각 메서드는 self.metadata_url, self.data_url 에 의존한다.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class _HttpMixin:
    """EdgeX HTTP 엔드포인트 통신 유틸리티 믹스인."""

    # ── HTTP 상태 설명 ────────────────────────────────────────────────────────

    def _describe_http_status(self, status_code: int) -> str:
        status_map = {
            200: "성공: 요청 처리 완료",
            201: "생성됨: 리소스 생성 성공",
            202: "수락됨: 요청 수락, 처리 중",
            204: "본문 없음: 성공적으로 처리됨",
            207: "복합 상태: 부분 성공/실패 혼재",
            400: "잘못된 요청: 데이터 형식 오류 또는 잘못된 데이터",
            401: "인증 실패: JWT 토큰 누락 또는 유효하지 않음",
            403: "권한 없음: 접근 권한 없음",
            404: "리소스 없음: 요청한 리소스를 찾을 수 없음",
            405: "허용되지 않은 메서드",
            408: "요청 시간 초과",
            409: "충돌: 리소스 충돌(이미 존재 등)",
            415: "지원되지 않는 콘텐츠 타입",
            422: "처리 불가 엔터티: 검증 실패",
            423: "잠금됨: 디바이스 잠금 또는 운영 상태 비활성화",
            429: "요청 과다",
            500: "내부 서버 오류",
            502: "게이트웨이 오류",
            503: "서비스 사용 불가 또는 연결 제한",
            504: "게이트웨이 시간 초과",
        }
        return status_map.get(status_code, "알 수 없는 오류")

    # ── 엔드포인트 생성 ───────────────────────────────────────────────────────

    def _versioned_endpoints(
        self,
        base_url: str,
        resource_path: str,
        include_legacy: bool = False,
    ) -> List[str]:
        """버전별 엔드포인트 목록 생성 (v3 → v2 → v1 → 레거시)."""
        base = base_url.rstrip("/")
        resource = resource_path.lstrip("/")
        endpoints = [
            f"{base}/api/v3/{resource}",
            f"{base}/api/v2/{resource}",
            f"{base}/api/v1/{resource}",
        ]
        if include_legacy:
            endpoints.append(f"{base}/{resource}")
        return endpoints

    def _payload_for_endpoint(
        self, endpoint: str, payload: Dict[str, object]
    ) -> object:
        """엔드포인트 버전에 따라 페이로드 형식 조정 (v3=배열, v2/v1=객체)."""
        return [payload] if "/v3/" in endpoint else payload

    # ── 응답 파싱 ─────────────────────────────────────────────────────────────

    def _extract_multistatus_item(
        self, response: requests.Response
    ) -> Optional[Dict[str, object]]:
        """207 복합 상태 응답에서 첫 번째 항목 추출."""
        try:
            result = response.json()
        except ValueError:
            return None
        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
            return result[0]
        return None

    def _response_status_code(self, response: requests.Response) -> int:
        """응답에서 실제 상태 코드 추출 (207 복합 상태 지원)."""
        if response.status_code != 207:
            return response.status_code
        item = self._extract_multistatus_item(response)
        if item and isinstance(item.get("statusCode"), int):
            return item["statusCode"]
        return response.status_code

    def _response_id(self, response: requests.Response) -> Optional[str]:
        """응답에서 생성된 리소스 ID 추출 (유연한 형식 지원)."""
        if response.status_code == 207:
            item = self._extract_multistatus_item(response)
            if item:
                value = item.get("id")
                return str(value) if value is not None else None

        try:
            data = response.json()
            if isinstance(data, dict):
                value = data.get("id")
                return str(value) if value is not None else None
        except ValueError:
            return None
        return None

    # ── 비동기 요청 래퍼 ─────────────────────────────────────────────────────

    async def _request_get(
        self, endpoint: str, timeout: int = 5
    ) -> Optional[requests.Response]:
        try:
            return await asyncio.to_thread(requests.get, endpoint, timeout=timeout)
        except requests.RequestException as error:
            logger.debug("GET 실패 (%s): %s", endpoint, error)
            return None

    async def _request_post(
        self,
        endpoint: str,
        payload: object,
        timeout: int = 10,
    ) -> Optional[requests.Response]:
        try:
            return await asyncio.to_thread(
                requests.post,
                endpoint,
                json=payload,
                timeout=timeout,
                headers={"Content-Type": "application/json"},
            )
        except requests.RequestException as error:
            logger.debug("POST 실패 (%s): %s", endpoint, error)
            return None

    async def _request_delete(
        self, endpoint: str, timeout: int = 10
    ) -> Optional[requests.Response]:
        try:
            return await asyncio.to_thread(requests.delete, endpoint, timeout=timeout)
        except requests.RequestException as error:
            logger.debug("DELETE 실패 (%s): %s", endpoint, error)
            return None

    # ── 고수준 요청 ──────────────────────────────────────────────────────────

    async def _post_with_versioned_fallback(
        self,
        base_url: str,
        resource_path: str,
        payload: Dict[str, object],
        operation_name: str,
    ) -> str:
        """v3 → v2 → v1 순서로 POST 를 시도하고 결과 문자열 반환."""
        endpoints = self._versioned_endpoints(base_url, resource_path)

        for endpoint in endpoints:
            request_payload = self._payload_for_endpoint(endpoint, payload)
            response = await self._request_post(endpoint, request_payload, timeout=10)
            if response is None:
                logger.debug("%s 실패: 응답 없음 (%s)", operation_name, endpoint)
                continue

            status_code = self._response_status_code(response)
            if status_code in [200, 201]:
                logger.info("✓ %s: %s", operation_name, endpoint)
                return "success"
            if status_code == 409:
                logger.info("✓ %s 이미 존재: %s", operation_name, endpoint)
                return "exists"
            if status_code == 404:
                logger.debug("엔드포인트 없음: %s", endpoint)
                continue
            if response.status_code == 207:
                logger.warning("%s 실패: 207 응답 - %s", operation_name, response.text)
                continue

            logger.warning(
                "%s 실패: %s - %s",
                operation_name, status_code, self._describe_http_status(status_code),
            )
            logger.debug("응답: %s", response.text)

        return "failed"

    async def _probe_service_health(self, base_url: str, service_name: str) -> bool:
        """EdgeX 서비스 헬스 체크 (ping 엔드포인트)."""
        endpoints = self._versioned_endpoints(base_url, "ping", include_legacy=True)

        for endpoint in endpoints:
            response = await self._request_get(endpoint, timeout=5)
            if response and response.status_code == 200:
                logger.info("✓ EdgeX %s 연결됨 (%s)", service_name, endpoint)
                return True

        logger.warning("EdgeX %s 연결 실패 - 시도한 엔드포인트:", service_name)
        for endpoint in endpoints:
            logger.warning("  - %s", endpoint)
        return False

    async def _post_event_via_rest(
        self,
        camera_id: str,
        event_type: str,
        base_event: Dict[str, object],
    ) -> bool:
        """Core Data REST 엔드포인트로 이벤트를 직접 POST."""
        endpoints = self._versioned_endpoints(self.data_url, "event")

        last_status = None
        last_text = None
        last_endpoint = None

        for endpoint in endpoints:
            last_endpoint = endpoint
            api_version = "v3" if "/v3/" in endpoint else ("v2" if "/v2/" in endpoint else "v1")
            event_data = {"apiVersion": api_version, **base_event}
            payload = [event_data] if api_version == "v3" else event_data

            response = await self._request_post(endpoint, payload, timeout=10)
            if response is None:
                continue

            status_code = self._response_status_code(response)
            last_status = status_code
            last_text = response.text

            if status_code in [200, 201]:
                logger.debug("[%s] EdgeX 이벤트 전송: %s", camera_id, event_type)
                return True
            if status_code == 404:
                logger.warning("엔드포인트 없음: %s", endpoint)
                continue
            if response.status_code == 207:
                logger.warning("Event 전송 실패 (%s): 207 응답 - %s", camera_id, response.text)
                continue

            logger.warning(
                "Event 전송 실패 (%s): %s - %s",
                camera_id, status_code, self._describe_http_status(status_code),
            )
            logger.warning("응답: %s / 엔드포인트: %s", response.text, endpoint)

        logger.warning("이벤트 전송 실패 (%s) - 모든 엔드포인트 시도 완료", camera_id)
        if last_endpoint:
            logger.warning("마지막 엔드포인트: %s", last_endpoint)
        if last_status is not None:
            logger.warning(
                "마지막 상태 코드: %s - %s",
                last_status, self._describe_http_status(last_status),
            )
        if last_text:
            logger.warning("마지막 응답: %s", last_text)

        return False

    # ── 범용 엔터티 CRUD ─────────────────────────────────────────────────────

    async def _get_entity_by_name(
        self,
        resource_path: str,
        container_key: str,
    ) -> Optional[Dict[str, object]]:
        """이름으로 EdgeX 엔터티 조회 (성공 시 반환)."""
        endpoints = self._versioned_endpoints(self.metadata_url, resource_path)

        for endpoint in endpoints:
            response = await self._request_get(endpoint, timeout=5)
            if response is None:
                logger.debug("엔터티 조회 응답 없음: %s", endpoint)
                continue
            try:
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, dict):
                        nested = data.get(container_key)
                        if isinstance(nested, dict):
                            return nested
                        return data
                if response.status_code == 404:
                    continue
            except Exception:
                continue

        return None

    async def _delete_entity_by_name(
        self,
        resource_path: str,
        success_log: Optional[str] = None,
    ) -> bool:
        """이름으로 EdgeX 엔터티 삭제 (성공 시 로그 기록)."""
        endpoints = self._versioned_endpoints(self.metadata_url, resource_path)

        for endpoint in endpoints:
            response = await self._request_delete(endpoint, timeout=10)
            if response is None:
                logger.debug("엔터티 삭제 응답 없음: %s", endpoint)
                continue
            try:
                if response.status_code in [200, 202, 204]:
                    if success_log:
                        logger.info(success_log)
                    return True
                if response.status_code == 404:
                    return True
            except Exception:
                continue

        return False

"""API v1 라우터 패키지.

필요한 서브모듈만 직접 import 하도록 유지한다.
패키지 import 시 전체 라우터를 eager import 하면 테스트와 CLI 스크립트에서
불필요한 초기화 비용이 커지고, 무거운 의존성이 같이 로드될 수 있다.
"""

__all__ = [
    "alerts",
    "appearances",
    "cameras",
    "control",
    "event_reviews",
    "events",
    "health",
    "metrics",
    "search",
    "sites",
]

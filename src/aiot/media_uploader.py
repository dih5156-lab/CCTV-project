from __future__ import annotations

import hashlib
import mimetypes
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional
from urllib.parse import urlparse

import requests

from src.aiot.contracts import FetchMediaRequest


class MediaUploadError(RuntimeError):
    """요청 미디어를 안전하게 업로드할 수 없을 때 발생한다."""


@dataclass(frozen=True)
class UploadResult:
    match_id: str
    bytes_uploaded: int
    sha256: str
    status: str = "completed"


class MediaUploader:
    def __init__(
        self,
        *,
        allowed_hosts: Iterable[str],
        media_roots: Iterable[Path | str],
        session: Optional[requests.Session] = None,
        timeout_seconds: float = 30.0,
    ):
        self.allowed_hosts = {host.strip().lower() for host in allowed_hosts if host.strip()}
        self.media_roots = tuple(Path(root).resolve() for root in media_roots)
        self.session = session or requests.Session()
        self.timeout_seconds = timeout_seconds

    def upload(
        self,
        request: FetchMediaRequest,
        resolve_match: Callable[[str], Optional[Path]],
    ) -> list[UploadResult]:
        self._validate_request(request)
        match_id = request.match_ids[0]
        resolved = resolve_match(match_id)
        if resolved is None:
            raise MediaUploadError("requested media was not found")
        file_path = Path(resolved).resolve()
        if not any(file_path.is_relative_to(root) for root in self.media_roots):
            raise MediaUploadError("resolved media path is outside allowed roots")
        if not file_path.is_file():
            raise MediaUploadError("requested media was not found")
        self._validate_kind(file_path, request.media_kind)
        size = file_path.stat().st_size
        if size > request.max_bytes:
            raise MediaUploadError("requested media exceeds max_bytes")

        digest = hashlib.sha256()
        with file_path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        with file_path.open("rb") as source:
            response = self.session.put(
                request.upload_url,
                data=source,
                headers={"Content-Type": content_type, "Content-Length": str(size)},
                timeout=self.timeout_seconds,
            )
        response.raise_for_status()
        return [UploadResult(match_id, size, digest.hexdigest())]

    def _validate_request(self, request: FetchMediaRequest) -> None:
        parsed = urlparse(request.upload_url)
        if parsed.scheme.lower() != "https":
            raise MediaUploadError("upload URL must use HTTPS")
        host = (parsed.hostname or "").lower()
        if host not in self.allowed_hosts:
            raise MediaUploadError("upload host is not allowed")
        if request.expires_at <= datetime.now(timezone.utc):
            raise MediaUploadError("upload URL expired")

    @staticmethod
    def _validate_kind(file_path: Path, media_kind: str) -> None:
        suffix = file_path.suffix.lower()
        allowed = {
            "snapshot": {".jpg", ".jpeg", ".png", ".webp"},
            "clip": {".mp4", ".mkv", ".webm"},
        }
        if suffix not in allowed[media_kind]:
            raise MediaUploadError("media extension does not match media_kind")


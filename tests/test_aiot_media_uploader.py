from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.aiot.contracts import FetchMediaRequest
from src.aiot.media_uploader import MediaUploadError, MediaUploader


class FakeResponse:
    status_code = 200

    def raise_for_status(self):
        return None


class FakeSession:
    def __init__(self):
        self.calls = []

    def put(self, url, data, headers, timeout):
        self.calls.append((url, data.read(), headers, timeout))
        return FakeResponse()


def _request(url="https://uploads.example.com/object"):
    return FetchMediaRequest(
        request_id="m-1",
        parent_request_id="q-1",
        match_ids=("match-1",),
        media_kind="snapshot",
        upload_url=url,
        max_bytes=1024,
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
    )


def test_rejects_upload_host_outside_allowlist(tmp_path):
    uploader = MediaUploader(
        session=FakeSession(), allowed_hosts={"allowed.example.com"}, media_roots=[tmp_path]
    )
    with pytest.raises(MediaUploadError, match="host"):
        uploader.upload(_request(), lambda _: tmp_path / "event.jpg")


def test_rejects_resolved_path_outside_media_roots(tmp_path):
    outside = tmp_path.parent / "secret.jpg"
    outside.write_bytes(b"secret")
    uploader = MediaUploader(
        session=FakeSession(), allowed_hosts={"uploads.example.com"}, media_roots=[tmp_path]
    )
    with pytest.raises(MediaUploadError, match="outside"):
        uploader.upload(_request(), lambda _: outside)


def test_uploads_only_requested_match_and_returns_checksum(tmp_path):
    media = tmp_path / "event.jpg"
    media.write_bytes(b"image-bytes")
    session = FakeSession()
    uploader = MediaUploader(
        session=session, allowed_hosts={"uploads.example.com"}, media_roots=[tmp_path]
    )

    results = uploader.upload(_request(), lambda match_id: media if match_id == "match-1" else None)

    assert len(session.calls) == 1
    assert session.calls[0][1] == b"image-bytes"
    assert results[0].match_id == "match-1"
    assert results[0].bytes_uploaded == len(b"image-bytes")
    assert len(results[0].sha256) == 64


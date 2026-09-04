#!/usr/bin/env python3
"""Record a fall demonstration video and API events in one session directory."""

from __future__ import annotations

import argparse
import getpass
import json
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

DEFAULT_OVERLAY_SOURCE = "rtsp://127.0.0.1:8554/sample_eval"
DEFAULT_SOURCE = "rtsp://192.168.0.100:554/stream1"


def _session_dir(root: Path) -> Path:
    name = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = root / name
    path.mkdir(parents=True, exist_ok=False)
    return path


def _read_env_key(path: Path) -> str | None:
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("PUBLIC_API_KEY="):
            return line.split("=", 1)[1].strip().strip("\"'") or None
    return None


def _fetch_events(api_url: str, camera_id: str | None, api_key: str | None) -> list[dict]:
    query = {"limit": "100"}
    if camera_id:
        query["camera_id"] = camera_id
    url = f"{api_url.rstrip('/')}/api/v1/events?{urlencode(query)}"
    headers = {"Accept": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    request = Request(url, headers=headers)
    with urlopen(request, timeout=5) as response:  # noqa: S310 - user supplies local API
        payload = json.loads(response.read().decode("utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    events = []
    if isinstance(payload, dict):
        events = payload.get("events") or payload.get("items") or []
    return [item for item in events if isinstance(item, dict)]


def _event_worker(
    api_url: str,
    camera_id: str | None,
    api_key: str | None,
    output_path: Path,
    stop: threading.Event,
    interval: float,
) -> None:
    seen: set[str] = set()
    with output_path.open("w", encoding="utf-8") as stream:
        while not stop.is_set():
            try:
                events = _fetch_events(api_url, camera_id, api_key)
                for event in reversed(events):
                    key = str(event.get("event_id") or event.get("id") or json.dumps(event, sort_keys=True))
                    if key in seen:
                        continue
                    seen.add(key)
                    record = {
                        "captured_at": datetime.now(timezone.utc).isoformat(),
                        "event": event,
                    }
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
            except Exception as exc:  # keep recording if API briefly drops
                stream.write(json.dumps({"captured_at": datetime.now(timezone.utc).isoformat(), "error": str(exc)}, ensure_ascii=False) + "\n")
                stream.flush()
            stop.wait(interval)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE, help=f"RTSP/file source (default: {DEFAULT_SOURCE})")
    parser.add_argument(
        "--overlay-source",
        default=DEFAULT_OVERLAY_SOURCE,
        help=f"Processed RTSP source with AI overlays (default: {DEFAULT_OVERLAY_SOURCE})",
    )
    parser.add_argument("--api-url", default="http://127.0.0.1:9000")
    parser.add_argument("--camera-id")
    parser.add_argument("--api-key", help="Public API key (or set PUBLIC_API_KEY)")
    parser.add_argument("--env-file", type=Path, default=Path(".env.jetson"), help="Environment file used for PUBLIC_API_KEY")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--output-root", type=Path, default=Path("data/fall_demo"))
    parser.add_argument("--poll-interval", type=float, default=1.0)
    args = parser.parse_args()
    if args.duration <= 0 or args.poll_interval <= 0:
        parser.error("--duration and --poll-interval must be positive")

    session = _session_dir(args.output_root)
    video_path = session / "demo.mp4"
    events_path = session / "events.jsonl"
    metadata = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "source": args.source,
        "api_url": args.api_url,
        "camera_id": args.camera_id,
        "duration_seconds": args.duration,
    }
    (session / "session.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    stop = threading.Event()
    import os
    api_key = args.api_key or os.environ.get("PUBLIC_API_KEY") or _read_env_key(args.env_file)
    if not api_key:
        api_key = getpass.getpass("Public API key (hidden, optional): ").strip() or None
    worker = threading.Thread(target=_event_worker, args=(args.api_url, args.camera_id, api_key, events_path, stop, args.poll_interval), daemon=True)
    worker.start()
    def build_ffmpeg_command(source: str, output: Path, preserve_timestamps: bool = False) -> list[str]:
        command = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y"]
        if source.startswith("rtsp://"):
            command += ["-rtsp_transport", "tcp"]
        timing = ["-vsync", "0"] if preserve_timestamps else ["-vf", "fps=30", "-r", "30", "-vsync", "2"]
        return command + ["-fflags", "+genpts", "-i", source, "-t", str(args.duration), "-map", "0:v:0"] + timing + [
            "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
        ]

    processes = [subprocess.Popen(build_ffmpeg_command(args.source, video_path))]
    overlay_path = session / "overlay.mp4"
    if args.overlay_source:
        processes.append(subprocess.Popen(build_ffmpeg_command(args.overlay_source, overlay_path, preserve_timestamps=True)))
    try:
        return_codes = [process.wait() for process in processes]
    finally:
        stop.set()
        worker.join(timeout=3)
    metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
    metadata["ffmpeg_returncode"] = return_codes[0]
    if args.overlay_source:
        metadata["overlay_source"] = args.overlay_source
        metadata["overlay_ffmpeg_returncode"] = return_codes[1]
    (session / "session.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"session: {session}")
    print(f"video:   {video_path}")
    if args.overlay_source:
        print(f"overlay: {overlay_path}")
    print(f"events:  {events_path}")
    return max(return_codes)


if __name__ == "__main__":
    raise SystemExit(main())

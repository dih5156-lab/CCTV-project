"""TLS-verified storage lookup; never interpret arbitrary HTTP errors as absence."""

import json
import math
from pathlib import Path
from urllib.parse import quote, urlsplit

import requests

from ..monitoring_contract import MonitoringContractError, require_string


class HttpMonitoringReader:
    def __init__(self, base_url, token, ca_file):
        parsed = urlsplit(base_url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in ("", "/")
        ):
            raise ValueError(
                "An HTTPS origin without credentials/path/query is required"
            )
        if (
            not isinstance(token, str)
            or len(token) < 32
            or not token.isascii()
            or any(c.isspace() for c in token)
        ):
            raise ValueError("A valid API token is required")
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.ca_file = str(Path(ca_file).resolve())
        if not Path(self.ca_file).is_file():
            raise ValueError("An explicit CA certificate file is required")
        self.endpoint = json.dumps(
            {"kind": "https-monitoring-status", "base_url": self.base_url},
            sort_keys=True,
        )

    def get_delivery_status(self, binding, delivery_id, *, timeout):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("positive finite timeout required")
        require_string(delivery_id)
        with requests.Session() as session:
            session.trust_env = False
            with session.get(
                self.base_url + "/v1/deliveries/" + quote(delivery_id, safe=""),
                params=binding.as_dict(),
                headers={"Authorization": "Bearer " + self.token},
                verify=self.ca_file,
                timeout=timeout,
                allow_redirects=False,
                stream=True,
            ) as response:
                if response.status_code == 401:
                    raise MonitoringContractError("unauthorized")
                if response.status_code == 403:
                    raise MonitoringContractError("binding_mismatch")
                if response.status_code not in (200, 404):
                    raise ConnectionError("monitoring_status_unavailable")
                data = bytearray()
                for chunk in response.iter_content(4096):
                    data.extend(chunk)
                    if len(data) > 65536:
                        raise MonitoringContractError("ack_mismatch")
                try:
                    body = json.loads(data)
                except (ValueError, UnicodeError, RecursionError):
                    raise MonitoringContractError("ack_mismatch") from None
                if response.status_code == 404:
                    if body != {
                        "status": "not_found",
                        "binding": binding.as_dict(),
                        "delivery_id": delivery_id,
                    }:
                        raise MonitoringContractError("ack_mismatch")
                    return None
                if not isinstance(body, dict):
                    raise MonitoringContractError("ack_mismatch")
                return body

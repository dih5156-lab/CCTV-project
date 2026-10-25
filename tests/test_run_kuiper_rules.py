import json
from pathlib import Path
from types import SimpleNamespace

from runners import run_kuiper_rules


def test_rule_deployment_retries_temporary_broker_failure(monkeypatch):
    requests = []

    def request(method, url, **kwargs):
        requests.append((method, url, kwargs))
        return SimpleNamespace(status_code=404 if method == "GET" else 201, text="")

    monkeypatch.setattr(run_kuiper_rules, "_request", request)
    rule = {"id": "retry_check", "sql": "SELECT * FROM input", "actions": [{"log": {}}]}
    run_kuiper_rules.upsert_rule("http://kuiper", rule)
    payload = requests[-1][2]["json"]

    # 브로커의 일시적인 기동 지연을 허용하되 재시도 횟수는 제한한다.
    assert payload["options"]["restartStrategy"] == {
        "attempts": 20, "delay": 1000, "maxDelay": 30000,
        "multiplier": 2, "jitter": 0.1,
    }
    assert {key: payload[key] for key in rule} == rule


def test_single_event_rules_preserve_source_event_id():
    # 투영에서 원본 ID를 잃으면 Action Layer가 경로마다 다른 명령 ID를 만든다.
    definition = json.loads((Path(__file__).resolve().parents[1] /
                             "kuiper/rules/cctv_intrusion_rules.json").read_text())
    assert "event_id STRING" in definition["stream"]["sql"]
    for rule in definition["rules"]:
        if rule["id"] in {"intrusion_confidence_filter", "intrusion_high_confidence_routing"}:
            assert "event_id" in rule["sql"].split(" FROM ")[0]


def test_ai_rules_publish_qos1_for_offline_action_session():
    # 수신 측의 지속 세션만 바꾸어도 QoS0 발행은 오프라인 구간에 보존되지 않는다.
    definition = json.loads((Path(__file__).resolve().parents[1] /
                             "kuiper/rules/cctv_intrusion_rules.json").read_text())
    for rule in definition["rules"]:
        assert all(action["mqtt"]["qos"] == 1 for action in rule["actions"])

import importlib.util
import json
import os
import subprocess
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
AM_DIR = ROOT / "monitoring" / "alertmanager"
WEBHOOK = {
    "version": "4",
    "alerts": [
        {
            "status": "firing",
            "labels": {"alertname": "TritonDown", "severity": "page"},
            "annotations": {"summary": "Triton is down"},
            "startsAt": "2026-01-01T00:00:00Z",
        }
    ],
}


@pytest.fixture
def sink():
    spec = importlib.util.spec_from_file_location("alert_sink", ROOT / "monitoring" / "alert-sink" / "sink.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.received.clear()
    server = ThreadingHTTPServer(("127.0.0.1", 0), mod.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST")
    return urllib.request.urlopen(req)


def test_sink_logs_each_alert_and_serves_them_back(sink, capsys):
    assert json.load(post(f"{sink}/alerts", WEBHOOK)) == {"received": 1}
    (got,) = json.load(urllib.request.urlopen(f"{sink}/alerts"))
    assert (got["alertname"], got["severity"], got["status"]) == ("TritonDown", "page", "firing")
    line = json.loads(capsys.readouterr().out.strip())
    assert line["event"] == "alert" and line["summary"] == "Triton is down"


def test_sink_rejects_malformed_payload(sink):
    with pytest.raises(urllib.error.HTTPError) as e:
        post(f"{sink}/alerts", {"nope": 1})
    assert e.value.code == 400
    assert json.load(urllib.request.urlopen(f"{sink}/alerts")) == []


def render(tmp_path, slack_url):
    env = {k: v for k, v in os.environ.items() if k != "SLACK_WEBHOOK_URL"}
    if slack_url is not None:
        env["SLACK_WEBHOOK_URL"] = slack_url
    out = tmp_path / "am.yml"
    subprocess.run(["sh", str(AM_DIR / "render.sh"), "--check", str(out)], check=True, env=env)
    return yaml.safe_load(out.read_text()), tmp_path / "slack_webhook_url"


@pytest.mark.parametrize("unset", [None, ""])
def test_default_receiver_is_the_sink_only(tmp_path, unset):
    cfg, url_file = render(tmp_path, unset)
    (receiver,) = cfg["receivers"]
    assert receiver["name"] == cfg["route"]["receiver"] == "default"
    assert [w["url"] for w in receiver["webhook_configs"]] == ["http://alert-sink:8080/alerts"]
    assert "slack_configs" not in receiver and not url_file.exists()


def test_slack_added_only_when_webhook_set_and_url_kept_out_of_config(tmp_path):
    secret = "https://hooks.slack.invalid/services/T0/B0/very-secret"
    cfg, url_file = render(tmp_path, secret)
    (receiver,) = cfg["receivers"]
    assert receiver["webhook_configs"][0]["url"] == "http://alert-sink:8080/alerts"  # the sink stays
    (slack,) = receiver["slack_configs"]
    assert slack["api_url_file"] == str(url_file) and url_file.read_text() == secret
    assert secret not in (tmp_path / "am.yml").read_text()
    assert oct(url_file.stat().st_mode & 0o777) == "0o600"

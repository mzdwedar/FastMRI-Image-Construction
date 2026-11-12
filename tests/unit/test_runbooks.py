"""Every alert has a runbook page that exists, is linked by its rule, and has the four sections an on-call needs."""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
RUNBOOKS = ROOT / "docs" / "runbooks"
URL = "https://github.com/mzdwedar/FastMRI-Image-Construction/blob/main/docs/runbooks/{}.md"
SECTIONS = ("## Symptoms", "## Dashboards", "## Likely causes", "## Remediation")


def alert_rules() -> dict[str, dict]:
    rules = {}
    for path in sorted((ROOT / "monitoring" / "prometheus" / "rules").glob("*.yml")):
        for group in yaml.safe_load(path.read_text())["groups"]:
            for rule in group["rules"]:
                if "alert" in rule:
                    rules[rule["alert"]] = rule
    return rules


RULES = alert_rules()


def test_alerts_were_found():
    assert len(RULES) >= 11


@pytest.mark.parametrize("name", sorted(RULES))
def test_alert_links_to_an_existing_runbook_with_all_sections(name):
    assert RULES[name]["annotations"]["runbook_url"] == URL.format(name)
    page = RUNBOOKS / f"{name}.md"
    assert page.is_file(), f"missing {page.relative_to(ROOT)}"
    text = page.read_text()
    assert text.startswith(f"# {name}"), "first line must be the alert name"
    for section in SECTIONS:
        assert section in text, f"{page.name}: missing '{section}'"


def test_every_alert_in_a_runbook_dashboard_link_uses_a_provisioned_uid():
    known = {"fastmri-service", "fastmri-triton", "fastmri-infra"}
    for page in RUNBOOKS.glob("*.md"):
        for chunk in page.read_text().split("localhost:3000/d/")[1:]:
            assert chunk.split("?")[0].split(")")[0].split(" ")[0].split("`")[0] in known, page.name


def test_no_runbook_without_an_alert():
    orphans = {p.stem for p in RUNBOOKS.glob("*.md")} - set(RULES)
    assert not orphans, f"runbooks with no alert: {sorted(orphans)}"

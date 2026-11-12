# AlertmanagerNotifyFailing

Severity: ticket. Alertmanager has failed to deliver notifications for an integration for 5 minutes. Alerts may be firing with nobody told.

## Symptoms

- `alertmanager_notifications_failed_total` rises for `webhook` (alert-sink) or `slack`.

## Dashboards

Alertmanager UI (`localhost:9093`).

## Likely causes

- Slack webhook URL wrong, revoked or unreachable (outbound network).
- `alert-sink` container down.

## Remediation

```
docker compose logs --tail 100 alertmanager | grep -i -E "notify|error"
docker compose ps alert-sink
```
Fix the webhook (`SLACK_WEBHOOK_URL` in the environment, then `docker compose up -d alertmanager`) or restart `alert-sink`. Meanwhile check `localhost:9090/alerts` by hand.

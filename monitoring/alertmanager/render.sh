#!/bin/sh
# Alertmanager does not expand environment variables in its config, so the container entrypoint renders it.
#   render.sh              render to /tmp/alertmanager.yml, then exec alertmanager
#   render.sh --check OUT  render to OUT only (used by `make check-configs`; URL file goes next to OUT)
set -eu
# The image ships its own /etc/alertmanager/alertmanager.yml, so the config is always read from next to this script.
src="$(dirname "$0")/alertmanager.yml"
if [ "${1:-}" = "--check" ]; then out=$2; else out=/tmp/alertmanager.yml; fi
url_file="$(dirname "$out")/slack_webhook_url"

cp "$src" "$out"
if [ -n "${SLACK_WEBHOOK_URL:-}" ]; then
  umask 077
  printf '%s' "$SLACK_WEBHOOK_URL" > "$url_file"   # read via api_url_file, so the URL never appears in the config
  cat >> "$out" <<YAML
    slack_configs:
      - api_url_file: $url_file
        send_resolved: true
        title: '[{{ .Status | toUpper }}] {{ .CommonLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ "\n" }}{{ end }}'
YAML
fi
[ "${1:-}" = "--check" ] && exit 0
exec /bin/alertmanager --config.file="$out" --storage.path=/alertmanager

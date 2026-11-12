# LatencySLOBurnSlow

Severity: ticket. A route is spending its latency error budget more than 6x too fast over both 6 h and 30 m. Same SLOs as `LatencySLOBurnFast`, just slower and steadier.

## Symptoms

- Latency for the named `route` is mildly but persistently above threshold.
- Nobody needs to be woken; the budget will be gone in about five days at this rate.

## Dashboards

[Service overview](http://localhost:3000/d/fastmri-service), [Triton](http://localhost:3000/d/fastmri-triton).

## Likely causes

- Capacity creeping up against the available CPUs.
- A slow regression after a deploy (model, image, or `GATEWAY_WORKERS`).
- A threshold that is too tight for this hardware (Q12 numbers came from one machine with an untrained model).

## Remediation

Follow the steps in `LatencySLOBurnFast`, but there is time to look at traces and compare with the last deploy. If the thresholds are simply wrong for production hardware, propose new numbers: changing an SLO needs approval (SPEC-observability, Boundaries).
